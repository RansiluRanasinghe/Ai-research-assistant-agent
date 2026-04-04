from retriever import RAGPipeline
from llm import LLMService

class Agent:

    def __init__(self, rag_pipeline, llm_service):
        self.rag = rag_pipeline
        self.llm_service = llm_service