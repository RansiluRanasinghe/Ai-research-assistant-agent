import os
import requests
import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

class LLMService:

    def __init__(self, model_name = "llama3.2:1b", use_api_fallback=True):
        self.modle_name = model_name
        self.use_api_url = "http://localhost:11434/api/generate"


    def generate(self, prompt, max_new_tokens=500):

        payload = {
            "model" : self.modle_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens
            }
        }

        try:
            response = requests.post(self.use_api_url, json=payload, timeout=10)    
            response.raise_for_status()

            return response.json()["response"]
        
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Is the Ollama app running in your system tray?"
        except Exception as e:
            return f"Error during generation: {str(e)}"
    
if __name__ == "__main__":
    llm = LLMService()
    test_prompt = "what is best path to learn Machine Learning?"
    print("Prompt:", test_prompt)
    print("Response:", llm.generate(test_prompt))