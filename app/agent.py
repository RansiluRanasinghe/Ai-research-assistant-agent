from retriever import RAGPipeline
from llm import LLMService

class Agent:

    def __init__(self, rag_pipeline, llm_service):
        self.rag = rag_pipeline
        self.llm_service = llm_service

    def run(self, query, memory_context=""):

        research_keywords = ["research", "paper", "document", "study", "according to", "what does the"]

        if any(kw in query.lower() for kw in research_keywords):
            return self.rag.generate_answer(query)
        else:
            prompt = f"{memory_context}\nAnswer the following question concisely and factually. If you don't know, say 'I don't know':\n{query}"
            return self.llm_service.generate(prompt, max_new_tokens=100)


if __name__ == "__main__":

    from unittest.mock import MagicMock

    mock_rag = MagicMock()
    mock_rag.generate_answer.return_value = "Mocked RAG answer."

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Mocked direct answer."

    agent = Agent(mock_rag, mock_llm)

    test_queries = [
        "What is the capital of France?",
        "Tell me about the research paper on Project Nebula."
    ]

    for q in test_queries:
        print(f"Query: {q}")
        print(f"Agent Response: {agent.run(q)}\n")            