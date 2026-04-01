import os
import numpy as np
import faiss
from .embeddings import EmbeddingModel
from .llm import LLMService
from .utils import load_document, chunk_text

class VectorStore:

    def __int__(self, embedding_model, dimension=384):
        
        self.emb_model = embedding_model
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

        