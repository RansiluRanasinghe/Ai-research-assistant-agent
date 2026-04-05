import streamlit as st
import os
from pathlib import Path
from llm import LLMService
from embeddings import EmbeddingModel
from retriever import RAGPipeline, VectorStore
from agent import Agent
from memory import Memory
from utils import load_document, chunks_text


st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="wide")
st.title("AI Research Assistant")

UPLOAD_DIR = "../user_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR = "../vector-store"

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