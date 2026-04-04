import  sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.llm import LLMService
from app.embeddings import EmbeddingModel
from app.retriever import RAGPipeline, VectorStore
from app.agent import Agent
from app.memory import Memory
from app.utils import load_document, chunk_text

def initialize_system(data_dir = "../data", index_dir = "../vector-store"):

    llm = LLMService()
    emb = EmbeddingModel()
    store = VectorStore(embedding_model=emb)

    if Path(index_dir).exists() and (Path(index_dir) / "index.faiss").exists():
        print("Loading existing vector store...")
        store.load(index_dir)
    else:
        print("Building vector store from documents...")
        docs = load_document(data_dir)
        if not docs:
            print("No documents found. The assistant will only answer general questions.")
        else:
            all_chunks = []
            for doc in docs:
                chunks = chunk_text(doc, chunk_size=300, overlap=50)
                all_chunks.extend(chunks)
            store.add_documents(all_chunks)
            store.save(index_dir)
            print(f"Vector store built with {len(all_chunks)} chunks.")

    rag = RAGPipeline(vector_store=store, llm_service=llm)
    agent = Agent(rag_pipeline=rag, llm_service=llm)
    memory = Memory(max_history=5)

    return agent, memory

def main():

    print("AI Research Assistant Agent: ")
    print("Type 'exit' to quit.\n")

    agent, memory = initialize_system()

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not query:
                print("Please enter a valid question.")
                continue

            response = agent.run(query)
            memory.add(query, response)

            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()            