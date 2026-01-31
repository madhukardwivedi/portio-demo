import streamlit as st
import os
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

# 1. GET API KEY (Safe Mode)
# Checks Streamlit Secrets first (Cloud), falls back to your hardcoded key (Local)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    # FALLBACK FOR LOCAL TESTING ONLY - REMOVE BEFORE PUBLIC SHARE IF PREFERRED
    os.environ[
        "OPENAI_API_KEY"] = "your api key"

INDEX_FOLDER = "faiss_index_local"
PDF_PATH = "Terms and Conditions_ Professional Notes.docx.pdf"


# --- HELPER FUNCTIONS ---
def clean_display_text(text):
    return text.replace("\n", " ").strip()


# --- INDEXING LOGIC (Runs once on Cloud Startup) ---
@st.cache_resource
def initialize_backend():
    # 1. Check if Index exists. If not, BUILD IT.
    if not os.path.exists(INDEX_FOLDER):
        if not os.path.exists(PDF_PATH):
            return None, f"❌ PDF file '{PDF_PATH}' not found in repo."

        with st.spinner("🚀 Building Index for the first time (this takes 1 min)..."):
            # Load PDF
            loader = PyPDFLoader(PDF_PATH)
            pages = loader.load()

            # Simple Cleaning & Splitting
            import re
            def clean_text(text):
                return text.replace("\n", " ").strip()

            documents = []
            # Split by Clause Regex
            full_text = "\n".join([p.page_content for p in pages])
            parts = re.split(r"(?m)(?=^\s*\d+(?:\.\d+)?\.\s+[A-Z])", full_text)

            for part in parts:
                if part.strip():
                    # Extract Heading
                    meta = "General Context"
                    match = re.match(r"^\s*(\d+(\.\d+)?\.\s+[A-Z].*?)(?=\s)", part)
                    if match: meta = match.group(1).strip()

                    documents.append(Document(
                        page_content=clean_text(part),
                        metadata={"clause": meta}
                    ))

            # Create Index
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(INDEX_FOLDER)

    # 2. Load the Index (Fast)
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
st.title("🏢 Investor Co-Pilot")
st.caption("AI Legal Auditor for 'Circles Andorra' Offering")

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Hello! I am your AI Auditor. I have indexed the Terms & Conditions. Ask me about **Limited Recourse** or **Bankruptcy**."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: What happens if the issuer goes bankrupt?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retriever, error = initialize_backend()

        if error:
            st.error(error)
            st.stop()

        docs = retriever.invoke(prompt)

        context_str = ""
        for i, d in enumerate(docs, 1):
            context_str += f"[{i}] SOURCE: {d.metadata.get('clause', 'Unknown')}\n{d.page_content}\n\n"

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        system_prompt = "You are a Legal Auditor. Answer using ONLY the context. Cite clauses."

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", f"Question: {prompt}\n\nContext:\n{context_str}")
        ])

        chain = rag_prompt | llm | StrOutputParser()
        response = chain.invoke({})

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.expander("🔎 Auditor View (Source Citations)"):
            for i, d in enumerate(docs):
                st.markdown(f"**{d.metadata.get('clause')}**")
                st.caption(clean_display_text(d.page_content)[:200] + "...")