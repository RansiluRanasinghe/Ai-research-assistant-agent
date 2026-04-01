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