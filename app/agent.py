from retriever import RAGPipeline
from llm import LLMService

class Agent:

    def __init__(self, rag_pipeline, llm_service):
        self.rag = rag_pipeline
        self.llm_service = llm_service

    def run(self, query):

        research_keywords = ["research", "paper", "document", "study", "according to", "what does the"]

        if any(kw in query.lower() for kw in research_keywords):
            return self.rag.generate_answer(query)
        else:
            prompt = f"Answer the following question concisely:\n{query}"
            return self.llm_service.generate(prompt, max_new_tokens=100)    