import os
from pathlib import Path
from pypdf import PdfReader

def load_document(directory):

    docs = []

    for file_path in Path(directory).iterdir():

        if file_path.suffix.lower() == ".pdf":
            
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "n".join(page.extract_text() for page in reader.pages)
                docs.append(text)

        elif file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(f.read())


    return docs

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
