# 👩‍💻 Member 2 — Data + RAG Engineer

**Branch**: `feature/member2-data-rag`
**Project**: Autonomous Multimodal Scientific Discovery Agent
**Role**: Owns the scientific knowledge base — data ingestion, processing, and retrieval.

> For the full project overview (all 5 members), see the [`main` branch README](https://github.com/Raji1674/Autonomous-Multimodal-Scientific-Discovery-Agent/blob/main/README.md).

---

## 🎯 Responsibility

This member builds the pipeline that feeds scientific knowledge to all other agents in the system.
The Literature Agent queries this pipeline to retrieve relevant papers when evaluating hypotheses.

---

## 🔄 Pipeline

```
arXiv API  ──┐
             ├──► Data Ingestion ──► PDF Processing ──► Chunking
PubMed API ──┘                           │
                                         ▼
                                     Embeddings
                                         │
                                         ▼
                                     Vector DB  (ChromaDB)
                                         │
                                         ▼
                              Semantic RAG Retrieval
                                         │
                                         ▼
                               Literature Agent (Member 1)
```

---

## 📁 Files Owned by Member 2

```
├── data/
│   ├── __init__.py
│   ├── arxiv_fetcher.py       ✅ Done — fetches papers from arXiv API
│   └── pubmed_fetcher.py      🔜 Next — fetches papers from PubMed/NCBI
│
├── rag/
│   ├── __init__.py
│   ├── cleaner.py             🔜 Strips HTML, normalises whitespace
│   ├── chunker.py             🔜 Splits text into overlapping chunks
│   ├── embedder.py            🔜 Generates sentence embeddings
│   └── vector_store.py        🔜 ChromaDB: store and query vectors
│
├── pipeline.py                🔜 End-to-end glue: fetch → chunk → embed → retrieve
├── requirements.txt           ✅ Done
└── .env.example               ✅ Done
```

---

## 🧱 Technology Stack

| Component | Library | Notes |
|-----------|---------|-------|
| arXiv API | `arxiv==2.1.3` | No API key needed |
| PubMed API | `biopython==1.84` | Requires `ENTREZ_EMAIL` in `.env` |
| PDF text extraction | `PyMuPDF==1.24.14` | `import fitz` |
| HTML cleaning | `beautifulsoup4==4.12.3` | Strip tags from abstracts |
| Text chunking | `langchain-text-splitters==0.3.8` | Recursive splitter, configurable overlap |
| Embeddings | `sentence-transformers==3.3.1` | Model: `all-MiniLM-L6-v2` (runs locally) |
| Vector database | `chromadb==0.5.23` | Local persistence, no server needed |
| Utilities | `python-dotenv`, `tqdm`, `requests` | Config, progress bars, HTTP |

---

## ⚙️ Setup for Member 2

### 1. Clone and switch to this branch
```bash
git clone https://github.com/Raji1674/Autonomous-Multimodal-Scientific-Discovery-Agent.git
cd Autonomous-Multimodal-Scientific-Discovery-Agent
git checkout feature/member2-data-rag
```

### 2. Create and activate virtual environment
```bash
# Create (Windows — use Python 3.10)
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
copy .env.example .env       # Windows
cp .env.example .env         # macOS / Linux
```
Then open `.env` and fill in:
```
ENTREZ_EMAIL=your_email@example.com   # any valid email, required for PubMed
```

---

## 🚀 Quick Test — arXiv Fetcher

After activating your venv:

```bash
python data/arxiv_fetcher.py
```

Expected output:
```
INFO: [arXiv] Searching for: 'large language models scientific discovery' (max 3 results)
INFO:   ✓ Fetched: A Comprehensive Survey of Scientific Large Language Models...
INFO:   ✓ Fetched: Hierarchical Multi-agent Large Language Model Reasoning...
INFO: [arXiv] Done. Fetched 3 papers.

============================================================
Fetched 3 papers from arXiv
============================================================

[1] A Comprehensive Survey of Scientific Large Language Models...
     Authors  : Yu Zhang, Xiusi Chen, Bowen Jin...
     Published: 2024-06-16
     ...
```

---

## 📊 Progress Tracker

| Step | Task | Status |
|------|------|--------|
| 1 | Project scaffold (requirements, .env, folders) | ✅ Done |
| 2 | arXiv fetcher (`data/arxiv_fetcher.py`) | ✅ Done — live tested |
| 3 | PubMed fetcher (`data/pubmed_fetcher.py`) | 🔜 In progress |
| 4 | Text cleaner (`rag/cleaner.py`) | ⬜ Pending |
| 5 | Chunker (`rag/chunker.py`) | ⬜ Pending |
| 6 | Embedder (`rag/embedder.py`) | ⬜ Pending |
| 7 | Vector store (`rag/vector_store.py`) | ⬜ Pending |
| 8 | End-to-end pipeline (`pipeline.py`) | ⬜ Pending |

---

## 🔑 Key Concepts (Beginner Reference)

| Term | Plain-English Explanation |
|------|---------------------------|
| **API** | A URL you call to get data back (like ordering from a menu) |
| **RAG** | Retrieval-Augmented Generation — give the AI real documents to read before answering |
| **Embedding** | Turning text into a list of numbers that captures its *meaning* |
| **Vector DB** | A database that finds documents with *similar meanings* (not just matching keywords) |
| **Chunk** | A small piece of a larger document (e.g. 500 words with 50-word overlap) |
| **Semantic Search** | Finding documents by meaning, not exact words |