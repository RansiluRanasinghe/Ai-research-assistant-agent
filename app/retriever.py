import os
import numpy as np
import faiss
from embeddings import EmbeddingModel
from llm import LLMService
from utils import load_document, chunk_text

import pickle
from rank_bm25 import BM25Okapi

class VectorStore:

    def __init__(self, embedding_model, dimension=384):
        
        self.emb_model = embedding_model
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.bm25 = None

    def add_documents(self, doc_texts):

        if not doc_texts:
            return

        #Faiss to vector
        vectors = self.emb_model.encode(doc_texts)
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(doc_texts)

        # BM25 for sparse retrieval
        tokenized_corpus = [doc.lower().split(" ") for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query, top_k=6):

        if not self.chunks:
            return []

        query_vec = self.emb_model.encode(query).astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))
        faiss_results = [self.chunks[i] for i in indices[0]]

        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_results = [self.chunks[i] for i in bm25_indices]

        combined_results = []
        seen = set()

        import itertools
        for f_chunk, b_chunk in itertools.zip_longest(faiss_results, bm25_results):
            if f_chunk and f_chunk not in seen:
                combined_results.append((f_chunk, 0.0))
                seen.add(f_chunk)

            if b_chunk and b_chunk not in seen:
                combined_results.append((b_chunk, 0.0))
                seen.add(b_chunk)    

        return combined_results[:top_k]

    def save(self, path):

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        np.save(os.path.join(path, "chunks.npy"), self.chunks)

        if self.bm25:
            with open(os.path.join(path, "bm25.pkl"), "wb") as f:
                pickle.dump(self.bm25, f)

    def load(self, path):

        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        self.chunks = np.load(os.path.join(path, "chunks.npy"), allow_pickle=True).tolist()
        self.dimension = self.index.d

        bm25_path = os.path.join(path, "bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)

class RAGPipeline:

    def __init__(self, vector_store, llm_service):
        self.vector_store = vector_store
        self.llm_service = llm_service 

    def generate_answer(self, query, top_k=6, max_new_tokens=300):

        retrieved = self.vector_store.hybrid_search(query, top_k=top_k)
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