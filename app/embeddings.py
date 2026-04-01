from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=32):

        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)


if __name__ == "__main__":
    emb = EmbeddingModel()

    test_texts = [
        "Hello, world!",
        "This is a test sentence.",
        "Another one for the vector."
    ]

    vectors = emb.encode(test_texts)
    print(f"Embedded {len(test_texts)} texts.")
    print(f"Vector shape: {vectors.shape}")
    print(f"First vector (first 5 dimensions): {vectors[0][:5]}")