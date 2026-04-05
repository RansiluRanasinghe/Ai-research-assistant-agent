import os
from pathlib import Path
from pypdf import PdfReader

def load_document(path):

    docs = []
    path_obj = Path(path)

    def process_file(file_path):

        if file_path.suffix.lower() == ".pdf":
            with open(file_path, "rb") as f:
                reader =PdfReader(f)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                return text

        elif file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()    

def chunk_text(text, chunk_size=1000, overlap=50):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":

    test_dir = Path("../data")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "sample.txt"
    test_file.write_text("This is a test document for the load_document function." * 50)

    docs = load_document(test_dir)
    print("Loaded Documents:", len(docs))

    if docs:
        print("\nFirst 200 characters of first document:")
        print(docs[0][:200])

        chunks = chunk_text(docs[0], chunk_size=50, overlap=10)
        print(f"\nCreated {len(chunks)} chunks.")
        print("First chunk:")
        print(chunks[0])
