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
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(doc_texts)

    def search(self, query, top_k=3):

        query_vec = self.emb_model.encode(query).astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = [(self.chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

        return results
    
    def save(self, path):

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        np.save(os.path.join(path, "chunks.npy"), self.chunks)

    def load(self, path):

        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        self.chunks = np.load(os.path.join(path, "chunks.npy"), allow_pickle=True).tolist()
        self.dimension = self.index.d    