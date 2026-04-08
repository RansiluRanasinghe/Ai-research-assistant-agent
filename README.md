<div align="center">

# 🤖 AI Research Assistant Agent

### Hybrid RAG · Local LLM · Agentic Web Routing · Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-00457C?style=flat-square)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

*An enterprise-grade, fully local AI research assistant — engineered from scratch. Retrieves knowledge using Hybrid Search (FAISS + BM25), guarantees transparency with source citations, and autonomously routes to the live internet when document knowledge falls short.*

</div>

---

## Overview

This project implements a **fully modular AI Research Assistant Agent** built on modern AI engineering principles, **completely bypassing heavy wrapper libraries like LangChain or LlamaIndex** to ensure deep architectural control.

Running **100% locally via Ollama**, this system combines **Hybrid Retrieval-Augmented Generation (RAG)** with an **agentic decision layer**. It intelligently decides whether to extract answers from your uploaded documents or pivot to real-time web scraping — all while providing transparent citations so you never have to blindly trust the AI.

> Building a RAG system with LangChain is straightforward. Building one entirely from scratch, without abstraction libraries, running fully offline — that's an architectural challenge. This project does exactly that.

---

## Features

| Feature | Description |
|---|---|
| 🧠 **Hybrid Search Retrieval** | Combines FAISS (Semantic Vector Search) with BM25 (Exact Keyword Match) to eliminate "keyword blindness" and retrieve highly accurate document chunks |
| 🌐 **Agentic Web Fallback** | A Smart Router that monitors FAISS distance scores — if a query falls outside the document's scope, the agent autonomously pivots to a live DuckDuckGo web search |
| 📚 **Transparent Citations** | Eliminates the "black box" problem. Every response includes a UI expander revealing the exact document paragraphs or live web URLs used to generate the answer |
| 🛠️ **Utility Action Engine** | Automated background prompts allowing users to instantly generate 1-page Summary Reports or extract statistical data into downloadable Markdown files |
| 🎨 **Reactive Streamlit UI** | A sleek, state-managed web interface featuring asynchronous processing, animated "Thinking" states, and secure document session management |
| 🔒 **100% Local Processing** | Uses `llama3.2:1b` via Ollama and local SentenceTransformers — no cloud APIs, no data leaks, fully offline-capable for document chatting |

---

## System Architecture

```
User Query ➔ Streamlit Web UI
                 │
                 ▼
          Agent (Smart Router)
                 │
     Is this a document research query?
     ├── [YES] ➔ Hybrid Retriever (FAISS Vectors + BM25 Keywords)
     │                 │
     │                 ▼
     │            Is the FAISS distance score strong?
     │            ├── [YES] ➔ Extract Context ──────────────┐
     │            └── [NO]  ➔ Trigger Web Fallback ─────────┤
     │                                                       │
     └── [NO]  ➔ DuckDuckGo Live Web Scraper ───────────────┘
                                                             │
                                                             ▼
                                          Ollama LLM (llama3.2) + Injected Context
                                                             │
                                                             ▼
                                      Final Answer + Transparent Source Citations
```

The Smart Router is the core differentiator. Rather than blindly querying the vector store on every request, the agent actively evaluates FAISS distance scores to determine retrieval confidence — then decides whether to trust the document index or escalate to a live web search.

---

## Core Components

**LLM Engine (Ollama)** — Runs `llama3.2:1b` entirely on-device. No API keys, no rate limits, no data leaving the machine. Swappable to larger 8B models with a single config change.

**Hybrid Retriever (FAISS + BM25)** — Dense vector search handles semantic understanding; sparse BM25 anchors exact keyword matches. Together they eliminate the "keyword blindness" problem that plagues pure vector search systems.

**Embedding Model** — `all-MiniLM-L6-v2` via SentenceTransformers, cached locally. Converts document chunks and queries into dense vector representations for similarity matching.

**Agentic Router** — The decision layer that monitors retrieval confidence scores in real time and autonomously selects between the document pipeline and the web fallback.

**Web Fallback (DuckDuckGo)** — When document knowledge is insufficient, the agent scrapes live search results. Implements a `lite` HTML backend fallback to handle rate-limiting gracefully.

**Memory Module** — Persists conversation history across turns to maintain coherent, context-aware multi-turn dialogue without re-indexing.

**Utility Action Engine** — Background prompt templates that allow one-click generation of Summary Reports and Statistical Data extractions, exported as downloadable Markdown files.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM Engine** | Ollama (`llama3.2:1b` — modular for 8B models) |
| **Embeddings** | `all-MiniLM-L6-v2` (SentenceTransformers, locally cached) |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Keyword Retrieval** | BM25 (`rank_bm25`) |
| **Web Search** | DuckDuckGo Search (`ddgs`) |
| **Document Processing** | `pypdf`, native Python chunking with overlap |
| **Runtime** | Python 3.10+, fully offline-capable |

---

## Project Structure

```
ai-research-assistant-agent/
│
├── app/
│   ├── web_app.py        # Streamlit UI & application state management
│   ├── agent.py          # Agent routing and web fallback logic
│   ├── retriever.py      # Hybrid RAG pipeline (FAISS + BM25)
│   ├── embeddings.py     # Local SentenceTransformer setup
│   ├── llm.py            # Local Ollama API interface
│   ├── memory.py         # Conversational context memory
│   └── utils.py          # PDF parsing & overlapping text chunker
│
├── user_uploads/         # Temporary storage for active session documents
├── vector-store/         # Persisted indices (index.faiss, chunks.npy, bm25.pkl)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Prerequisites

You must have [Ollama](https://ollama.com) installed. Once installed, pull the model:

```bash
ollama run llama3.2:1b
```

### 2. Clone the Repository

```bash
git clone https://github.com/RansiluRanasinghe/ai-research-assistant-agent.git
cd ai-research-assistant-agent
```

### 3. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

> Required packages: `streamlit`, `requests`, `sentence-transformers`, `faiss-cpu`, `rank_bm25`, `pypdf`, `ddgs`, `numpy`

### 4. Run the Application

```bash
cd app
streamlit run web_app.py
```

---

## Example Workflow

**1. Upload** — Drag and drop a research PDF into the sidebar. The UI locks and displays an animated status indicator while the document is chunked and indexed into FAISS and BM25.

**2. Hybrid Query** — Ask a specific question containing a technical acronym. Watch BM25 anchor the exact phrase while FAISS simultaneously retrieves semantic context — together eliminating retrieval gaps that either method alone would miss.

**3. Verify** — Click the `📚 View Source Context` expander below the AI's answer to audit the exact document chunks used to generate the response. Full transparency, no black box.

**4. Extract Data** — Click `Extract Statistical Data` in the sidebar to force the LLM to scan the entire PDF and generate a downloadable Markdown table of all numerical data.

**5. Test the Agent** — Ask `"What is the weather in Tokyo?"` — watch the agent recognize the PDF cannot answer this, bypass the vector database entirely, and return a live DuckDuckGo result with a web citation.

---

## Features in Action

**100% Local Document Ingestion**

![Local Ingestion UI](assests\ingestion-ui.png)

> Absolute Data Privacy: The reactive Streamlit interface safely parses, chunks, and indexes complex PDFs directly into a local vector store. By operating entirely on-device, the system guarantees zero data leaks and requires no external cloud APIs.

**Solving "Keyword Blindness"**

![Local Ingestion UI](assests\hybrid-search.png)

> Precision Hybrid Retrieval: Pure vector databases often struggle with exact terminology. This engine fuses the semantic understanding of FAISS with the exact-match precision of BM25, ensuring the AI never misses highly specific acronyms (like "DORA") or niche technical workflows.

**Zero-Tolerance for Hallucinations**

![Local Ingestion UI](assests\transparent-citations.png)

> Transparent Citations: Eliminating the "black box" of AI generation. Every response is securely anchored to an expandable UI citation block, allowing users to instantly verify the LLM's claims against the raw, extracted document chunks.

**Autonomous Agentic Web Routing**

![Local Ingestion UI](assests\web-fallback.png)

> Dynamic Fallback Logic: A custom decision engine continuously monitors FAISS distance scores. When a query falls outside the document's scope (e.g., real-time sports scores), the agent actively intercepts the request, bypasses the local index, and seamlessly scrapes DuckDuckGo to inject live web context.

## Key Engineering Highlights

✅ **Built a complete RAG architecture from scratch** — no LangChain, no LlamaIndex, no abstraction layers hiding the data flow.

✅ **Solved "Keyword Blindness"** — engineered a custom hybrid search algorithm fusing dense FAISS retrieval with sparse BM25 for coverage that neither achieves alone.

✅ **Designed an agentic confidence threshold** — the Smart Router evaluates FAISS distance scores at runtime to decide retrieval vs. web fallback, rather than blindly querying both.

✅ **Overcame web scraper rate-limiting** — implemented a graceful fallback to DuckDuckGo's `lite` HTML backend when the primary API is throttled.

✅ **Mastered Streamlit state management** — the chat interface stays responsive and thread-safe during heavy backend indexing operations.

✅ **Reduced hallucination via grounded generation** — every response is anchored to retrieved source chunks, with citations surfaced directly in the UI.

---

## Roadmap

- [x] ~~Web UI using Streamlit~~
- [x] ~~Agentic tool integration (live web search)~~
- [x] ~~Hybrid Search implementation (FAISS + BM25)~~
- [ ] Evaluation pipeline (ROUGE, BLEU, faithfulness metrics)
- [ ] GraphRAG integration for complex entity relationship mapping
- [ ] Multi-agent collaboration for advanced cross-document synthesis
- [ ] Dockerized deployment configuration

---

## Use Cases

- 📚 Research paper summarization and deep Q&A
- 🔬 Literature review and evidence extraction
- 🗂️ Custom private knowledge base querying
- 📝 Technical documentation assistant
- 📊 Statistical data extraction from dense reports

---

## Author

**Ransilu Ranasinghe** — Software Engineering Undergraduate | AI & Backend Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ransiluranasinghe)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)

---

<div align="center">

⭐ **Found this architecture useful? Give it a star and share your feedback!**

</div>