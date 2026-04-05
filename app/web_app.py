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