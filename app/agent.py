from retriever import RAGPipeline
from llm import LLMService

class Agent:

    def __init__(self, rag_pipeline, llm_service):
        self.rag = rag_pipeline
        self.llm_service = llm_service

    def run(self, query, memory_context="", max_new_tokens=300):

        research_keywords = ["research", "paper", "document", "study", "according", "what", "how", "why", "define", "explain", "who"]

        if any(kw in query.lower() for kw in research_keywords):
            return self.rag.generate_answer(query, max_new_tokens=max_new_tokens)
        else:
            prompt = f"Chat History:\n{memory_context}\n\nQuestion: {query}\nAnswer factually:"
            chat_limit = 100 if max_new_tokens == 300 else max_new_tokens
            answer = self.llm_service.generate(prompt, max_new_tokens=chat_limit)
            return answer.strip(), ""


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