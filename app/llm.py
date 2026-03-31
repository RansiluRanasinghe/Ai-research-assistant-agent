from transformers import pipeline

class LLMService:

    def __init__(self, model_name = "google/flan-t5-base"):
        self.pipeline = pipeline("text2text-generation", model=model_name, device=1)

    def generate(self, prompt, max_new_tokens=500):
        response = self.pipeline(prompt, max_new_tokens=max_new_tokens)
        return response[0]['generated_text']
    
if __name__ == "__main__":
    llm = LLMService()
    test_prompt = "what is best path to learn Machine Learning?"
    print("Prompt:", test_prompt)
    print("Response:", llm.generate(test_prompt))