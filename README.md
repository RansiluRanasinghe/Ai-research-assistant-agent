<div align="center">

# 🤖 AI Research Assistant Agent

### RAG · LLM · Tool-Augmented Reasoning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1ECB94?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-00457C?style=flat-square)](https://faiss.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

*An AI-powered research assistant that retrieves, understands, and summarizes knowledge from your own documents — grounded in source context, not hallucination.*

</div>

---

## Overview

This project implements a **fully modular AI Research Assistant Agent** built on modern LLM engineering principles. Unlike traditional chatbots that rely solely on parametric knowledge, this system combines **Retrieval-Augmented Generation (RAG)** with an **agent-based decision layer** to deliver accurate, source-grounded responses from external document collections.

The architecture is designed to be extensible — swap out any component (LLM, embeddings, vector store) without breaking the rest of the pipeline.

---

## Features

| Feature | Description |
|---|---|
| 📄 **Document Ingestion** | Load and process PDFs and plain text files |
| 🔍 **Semantic Search** | Vector similarity retrieval over embedded knowledge chunks |
| 🧠 **RAG Pipeline** | Context injection into LLM prompts for grounded responses |
| 🤖 **Agent Layer** | Decision logic for when to retrieve vs. respond directly |
| 💬 **Conversation Memory** | Maintains context across multi-turn interactions |
| 📊 **Source Attribution** | Every response is tied back to its source documents |

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Agent (Router)    │  ← Decides: retrieve or respond directly
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Retriever (FAISS)  │  ← Semantic search over vector store
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Context Injection  │  ← Relevant chunks added to prompt
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   LLM Generation    │  ← Response grounded in retrieved context
└─────────┬───────────┘
          │
          ▼
  Final Answer + Sources
```

---

## Core Components

**LLM Layer** — Natural language understanding and response generation via open-source models or the Gemini API.

**Embedding Model** — Converts text into dense vector representations using SentenceTransformers for semantic similarity matching.

**Vector Database (FAISS)** — Stores embeddings and enables fast, scalable nearest-neighbor search at query time.

**Retriever / RAG Pipeline** — Fetches the most relevant document chunks and injects them into the LLM context window.

**Agent Layer** — Routing logic that decides when to trigger retrieval versus answer from general knowledge directly.

**Memory Module** — Persists conversation history to maintain coherent, context-aware multi-turn dialogue.

---

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/-FAISS-00457C?style=flat-square)
![LangChain](https://img.shields.io/badge/-LangChain-1ECB94?style=flat-square&logo=chainlink&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/-LlamaIndex-7C3AED?style=flat-square)
![HuggingFace](https://img.shields.io/badge/-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Gemini](https://img.shields.io/badge/-Gemini_API-4285F4?style=flat-square&logo=google&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/-SentenceTransformers-FF6F00?style=flat-square)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

---

## Project Structure

```
ai-research-assistant-agent/
│
├── app/
│   ├── main.py           # Entry point (CLI interface)
│   ├── agent.py          # Agent routing and decision logic
│   ├── retriever.py      # RAG pipeline
│   ├── embeddings.py     # Embedding model setup
│   ├── llm.py            # LLM interface
│   ├── memory.py         # Conversation memory
│   └── utils.py          # Data loading & preprocessing
│
├── data/
│   ├── raw/              # Input documents (PDFs, text files)
│   └── processed/        # Chunked and cleaned text data
│
├── vector_store/
│   └── faiss_index/      # Persisted vector index
│
├── notebooks/
│   └── experiments.ipynb # Experimentation and prototyping
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-research-assistant-agent.git
cd ai-research-assistant-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root and add your credentials:

```env
HUGGINGFACE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### 4. Add Your Documents

Place your PDF or text files in the `data/raw/` directory.

### 5. Run the Assistant

```bash
python app/main.py
```

---

## How It Works

```
Step 1 │ INGEST      Documents are loaded and split into optimized chunks
Step 2 │ EMBED       Each chunk is converted to a vector via SentenceTransformers
Step 3 │ INDEX       Embeddings are stored in a FAISS vector index
Step 4 │ QUERY       User question is embedded and matched against the index
Step 5 │ RETRIEVE    Top-k relevant chunks are fetched
Step 6 │ GENERATE    LLM produces a response grounded in retrieved context
Step 7 │ ROUTE       Agent decides: retrieval path or direct response
Step 8 │ REMEMBER    Interaction stored in memory for conversational continuity
```

---

## Example Use Cases

- 📚 Research paper summarization and Q&A
- 🔬 Literature review assistance
- 🗂️ Custom knowledge base querying
- 📝 Technical documentation assistant

---

## Key Learnings

- ✅ Built a complete RAG pipeline from scratch
- ✅ Implemented vector search and semantic retrieval with FAISS
- ✅ Designed a modular, swappable agent architecture
- ✅ Integrated LLMs with external knowledge for grounded generation
- ✅ Reduced hallucination through context injection

---

## Roadmap

- [ ] Web UI using Streamlit or React
- [ ] Multi-document reasoning across large corpora
- [ ] Tool integration (live web search, external APIs)
- [ ] Evaluation pipeline (ROUGE, BLEU, faithfulness metrics)
- [ ] Dockerized deployment with cloud hosting

---

## Author

**Ransilu Ranasinghe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](www.linkedin.com/in/ransiluranasinghe)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)

---

<div align="center">

⭐ **Found this useful? Give it a star and share your feedback!**

</div>
