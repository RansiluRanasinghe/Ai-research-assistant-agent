import os
import requests
import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

class LLMService:

    def __init__(self, model_name = "google/flan-t5-small", use_api_fallback=True):
        self.modle_name = model_name
        self.use_api_fallback = use_api_fallback

        try:
            self.pipeline = pipeline("text2text-generation", model=self.modle_name, device= -1)

            self.local_available = True
        except Exception as e:
            print(f"Error loading local model: {e}")
            self.local_available = False

            self.api_token = os.getenv("HF_TOKEN")
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def generate(self, prompt, max_new_tokens=500):

        formatted_prompt = f"Answer the following question clearly:\n{prompt}"
        
        if self.local_available:
            try:
                response = self.pipeline(formatted_prompt, max_new_tokens=max_new_tokens)
                return response[0]['generated_text']
            except Exception as e:
                print("Local failed, switching to API:", e)

        if self.use_api_fallback:
            payload = payload = {
                "inputs": formatted_prompt,
                "parameters": {"max_new_tokens": max_new_tokens}
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)

            try:
                return response.json()[0]['generated_text']
            except Exception as e:
                return str(response.json)

        return "No model available"            
    
if __name__ == "__main__":
    llm = LLMService()
    test_prompt = "what is best path to learn Machine Learning?"
    print("Prompt:", test_prompt)
    print("Response:", llm.generate(test_prompt))