import streamlit as st
import os
import re
import shutil
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader

# --- CORRECT IMPORTS FOR CLOUD ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 2. Retrieval logic lives in the main 'langchain' package
try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except ModuleNotFoundError:
    # Fallback: Try langchain_community (sometimes they live here in newer builds)
    from langchain_community.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import CrossEncoderReranker

# --- CONFIG ---
st.set_page_config(page_title="Portio Investor Co-Pilot", page_icon="🏢", layout="wide")

# 1. GET API KEY (Safe Mode for Cloud)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    # Local Fallback
    os.environ["OPENAI_API_KEY"] = "your openai key"

INDEX_FOLDER = "faiss_index_local"
PDF_PATH = "Terms and Conditions_ Professional Notes.docx.pdf"

# --- HELPER: Normalize Text for Display ---
def clean_display_text(text):
    text = text.replace("\n", " ").strip()
    return re.sub(r'\s+', ' ', text)

# --- ROBUST METADATA EXTRACTOR (Restored) ---
def extract_clause_metadata(content):
    normalized = re.sub(r"\s+", " ", content).strip()
    # Matches: "9. EVENTS OF DEFAULT" (Number + Dot + Uppercase Title)
    m = re.match(r"^(\d+(?:\.\d+)?)\.\s*([A-Z][A-Z0-9 ,\-&()/]*?)(?=\s+[A-Z][a-z]|\s*$)", normalized)
    if m:
        return f"{m.group(1)}. {m.group(2).strip()}"
    return "General Context"

# --- INDEXING LOGIC (Runs once on Cloud Startup) ---
@st.cache_resource
def initialize_backend():
    # 1. Check if Index exists. If not, BUILD IT.
    if not os.path.exists(INDEX_FOLDER):
        if not os.path.exists(PDF_PATH):
             return None, f"❌ PDF file '{PDF_PATH}' not found in repo."
        
        with st.spinner("🚀 Building Index for the first time..."):
            loader = PyPDFLoader(PDF_PATH)
            pages = loader.load()
            
            documents = []
            full_text = "\n".join([p.page_content for p in pages])
            
            # Split by Clause Heading (Robust Regex)
            parts = re.split(r"(?m)(?=^\s*\d+(?:\.\d+)?\.\s+[A-Z][A-Z0-9 ,\-&()/]{2,}\s)", full_text)
            
            for part in parts:
                if part.strip():
                    meta = extract_clause_metadata(part)
                    documents.append(Document(
                        page_content=clean_display_text(part), 
                        metadata={"clause": meta}
                    ))
            
            # Create Index
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(INDEX_FOLDER)
    
    # 2. Load the Index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

    # 3. Setup Reranker
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 50})
    )
    return retriever, None

# --- UI LOGIC ---
st.title("Investor Co-Pilot")
st.caption("AI Legal Auditor for 'Circles Andorra' Offering")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Auditor. I have indexed the Terms & Conditions. Ask me about **Limited Recourse** or **Bankruptcy**."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: What happens if the issuer goes bankrupt?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal clauses..."):
            retriever, error = initialize_backend()
            
            if error:
                st.error(error)
                st.stop()

            # 1. Retrieve
            docs = retriever.invoke(prompt)
            
            # 2. Format Context
            context_str = ""
            for i, d in enumerate(docs, 1):
                context_str += f"[{i}] SOURCE: {d.metadata.get('clause', 'Unknown')}\n{d.page_content}\n\n"

            # 3. Generate Answer (UPDATED FOR VERBOSITY)
            llm = ChatOpenAI(model="gpt-4o", temperature=0) # Using GPT-4o for best logic
            
            system_prompt = """You are a meticulous Legal Auditor. Answer the user's question using ONLY the context provided.
            
            GUIDELINES:
            1. **Be Comprehensive:** Do not summarize. Explain the *interaction* between clauses (e.g., how Default relates to Deferral).
            2. **Cite & Quote:** Explicitly quote the text when possible (e.g., 'As stated in Clause 6: "..."').
            3. **Format:** Use paragraphs for readability. The user wants a detailed legal explanation, not a quick summary.
            """
            
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", f"Question: {prompt}\n\nContext:\n{context_str}")
            ])
            
            chain = rag_prompt | llm | StrOutputParser()
            response = chain.invoke({})
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # --- AUDITOR VIEW (TABULAR STYLE) ---
            with st.expander("🔎 Auditor View (Source Citations)", expanded=True):
                for i, d in enumerate(docs):
                    # This layout creates the clean "Table" look
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        # Bold Header (e.g., "6. INTEREST")
                        st.markdown(f"**{d.metadata.get('clause', 'N/A')}**")
                    
                    with col2:
                        # Clean Text Block
                        st.caption(d.page_content[:300] + "...")
                    
                    st.divider() # Adds a clean line between rows
