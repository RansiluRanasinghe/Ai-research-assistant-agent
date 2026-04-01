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

    def add_documents(self, doc_texts):

        if not doc_texts:
            return

        vectors = self.emb_model.encode(doc_texts)
        self.index.add(vectors.astype('float32'))
        self.chunks.extend(doc_texts)    