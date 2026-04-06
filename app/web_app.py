import shutil
import time

import streamlit as st
import os
from pathlib import Path
from llm import LLMService
from embeddings import EmbeddingModel
from retriever import RAGPipeline, VectorStore
from agent import Agent
from memory import Memory
from utils import load_document, chunk_text


st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="wide")
st.title("AI Research Assistant")

st.markdown("""
        <style>
            @keyframe pulse {
                0% {opacity: 1;}
                50% {opacity: 0.3;}
                100% {opacity: 1;}
            }

            .thinking-text{
                animation: pulse 1.5s infinite;
                font-style: italic;
                color: #888888;
            }    
        </style>
""", unsafe_allow_html=True)

UPLOAD_DIR = Path("../user_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR = Path("../vector-store")

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

@st.cache_resource
def init_core():

    llm = LLMService()
    emb = EmbeddingModel()
    store = VectorStore(embedding_model=emb)

    if(INDEX_DIR / "index.faiss").exists():
        store.load(str(INDEX_DIR))

    rag = RAGPipeline(vector_store=store, llm_service=llm)
    agent = Agent(rag_pipeline=rag, llm_service=llm)
    return agent, store

agent, vector_store = init_core()

if "memory" not in st.session_state:
    st.session_state.memory = Memory(max_history=10)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Document Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF or TXT research papers", type=["pdf", "txt"], accept_multiple_files=True)

    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):

                for uploaded_file in uploaded_files:
                    with open(UPLOAD_DIR / uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                docs = load_document(UPLOAD_DIR)
                all_chunks = []
                for doc in docs:
                    chunks = chunk_text(doc, chunk_size=150, overlap=30)
                    all_chunks.extend(chunks)

                vector_store.add_documents(all_chunks)
                vector_store.save(str(INDEX_DIR))
                st.success(f"Indexed {len(all_chunks)} new segments!")

        else:
            st.error("Please upload files first.")

    if st.button("Clear All Knowledge"):
        if UPLOAD_DIR.exists(): shutil.rmtree(UPLOAD_DIR)
        if INDEX_DIR.exists(): shutil.rmtree(INDEX_DIR)
        
        init_core.clear()
        st.session_state.memory = Memory(max_history=10)
        
        st.session_state.chat_history = []
        
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt :=st.chat_input("Ask a question about your documents..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = st.session_state.memory.get_context()

        think_placeholder = st.empty()
        think_placeholder.markdown("<div class='thinking-text'>Thinking... </div>", unsafe_allow_html=True)

        response = agent.run(prompt, memory_context=context)
        response = response.replace("Assistant:", "").replace("User:", "").strip()

        think_placeholder.empty()

        st.write_stream(stream_text(response))

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.memory.add(prompt, response)                                            