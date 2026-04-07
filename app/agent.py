from retriever import RAGPipeline
from llm import LLMService

from ddgs import DDGS

class Agent:

    def __init__(self, rag_pipeline, llm_service):
        self.rag = rag_pipeline
        self.llm_service = llm_service

    def run(self, query, memory_context="", max_new_tokens=300):

        research_keywords = ["research", "paper", "document", "study", "according", "what", "how", "why", "define", "explain", "who"]

        if any(kw in query.lower() for kw in research_keywords):

            answer, context, best_distance = self.rag.generate_answer(query, max_new_tokens=max_new_tokens)

            if best_distance > 1.2:
                print(f"Low document match score ({best_distance:.2f}). Falling back to Web Search...")
                return self.web_search(query, max_new_tokens)
            
            return answer.strip(), context
        
        else:
            prompt = f"Chat History:\n{memory_context}\n\nQuestion: {query}\nAnswer factually:"
            chat_limit = 100 if max_new_tokens == 300 else max_new_tokens
            answer = self.llm_service.generate(prompt, max_new_tokens=chat_limit)
            return answer.strip(), ""
        
    def web_search(self, query, max_new_tokens):
        print(f"\nInitiating Web Search for: '{query}'")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, backend="lite"))

            if not results:
                print("DDG 'lite' returned empty. Trying 'html' fallback...")
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))

            print(f"Successfully found {len(results)} web results.")        

            if not results:
                return "My web search was blocked by DuckDuckGo's anti-bot protections. Please try again in a few minutes!", ""       

            web_context =  "\n\n".join([f"Source: {res['title']}\n{res['body']}" for res in results])

            prompt = f"""You are a precise Research Assistant.
            Based on the following live web search results, answer the user's question.
            
            CRITICAL RULES:
            1. Only use factual, real-world information.
            2. IGNORE all references to video games, mods, forums, or fiction.
            3. Be concise and direct.
            
            Web Results:
            {web_context}
            
            Question: {query}
            Answer:"""

            answer =  self.llm_service.generate(prompt, max_new_tokens=max_new_tokens)
            return answer.strip(), f"PULLED FROM THE WEB:\n\n{web_context}"

        except Exception as e:
            print(f"Python Crash during web search: {e}")
            return f"Web search failed due to a system error.", ""  


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