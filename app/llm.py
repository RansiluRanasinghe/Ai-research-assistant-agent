from transformers import pipeline

class LLMService:

    def __init__(self, model_name = "google/flan-t5-base"):
        self.pipeline = pipeline("text2text-generation", model=model_name, device=1)