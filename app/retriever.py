import os
import numpy as np
import faiss
from embeddings import EmbeddingModel
from llm import LLMService
from utils import load_document, chunk_text

class VectorStore:

    def __init__(self, embedding_model, dimension=384):
        
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

        if not self.chunks:
            return []

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

class RAGPipeline:

    def __init__(self, vector_store, llm_service):
        self.vector_store = vector_store
        self.llm_service = llm_service 

    def generate_answer(self, query, top_k=6, max_new_tokens=300):

        retrieved = self.vector_store.search(query, top_k=top_k)
        context = "\n\n".join([chunk for chunk, _ in retrieved])

        prompt = f"""You are a precise Research Assistant. 
        Read the Context below. Answer the Question using ONLY the facts in the Context. 
        Be direct, concise, and do not add outside information.

        Context:
        {context}

        Question: {query}
        Answer:"""

        answer = self.llm_service.generate(prompt, max_new_tokens=max_new_tokens)

        return answer.strip(), context

if __name__ == "__main__":

    from pathlib import Path

    emb = EmbeddingModel()
    vector_store = VectorStore(emb)

    docs_dir = Path("../data")
    if docs_dir.exists():
        docs = load_document(docs_dir)
        all_chunks = []

        for doc in docs:
            chunks = chunk_text(doc, chunk_size=200, overlap=30)
            all_chunks.extend(chunks)

        vector_store.add_documents(all_chunks)
        print(f"Added {len(all_chunks)} chunks to the index.")

        query = "What is the main topic?"
        results = vector_store.search(query)
        print(f"Search results for '{query}':")

        for i, (chunk, dist) in enumerate(results):
            print(f"{i+1}. (dist={dist:.2f}) {chunk[:100]}...")

        llm = LLMService()
        rag = RAGPipeline(vector_store, llm)
        answer = rag.generate_answer(query)
        print(f"\nGenerated answer:\n{answer}")

    else:
        print(f"Documents directory '{docs_dir}' not found. Please add some documents to test the retriever.")        