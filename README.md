# 🔬 Autonomous Multimodal Scientific Discovery Agent

An AI-powered research assistant that autonomously retrieves scientific literature,
designs and runs ML experiments, critiques results statistically, and synthesises
findings into a human-readable research report — all driven by a multi-agent LLM pipeline.

---

## 🏗️ Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER (Member 1)              │
│          LangChain / AutoGen · LangGraph · State Management     │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────────────────────┐
│  Knowledge Base  │────▶│           AGENT PIPELINE             │
│   (Member 2)     │     │                                      │
│                  │     │  Literature Agent                    │
│  arXiv / PubMed  │     │       ↓                              │
│  PDF Processing  │     │  Hypothesis Agent                    │
│  Chunking        │     │       ↓                              │
│  Embeddings      │     │  Code Agent ──► Sandbox (Member 4)  │
│  Vector DB       │     │       ↓                              │
│  RAG Retrieval   │     │  Critic Agent   (Member 3)           │
└─────────────────┘     │       ↓                              │
                         │  Synthesis Agent (Member 5)           │
┌─────────────────┐     └──────────────────────────────────────┘
│  ML Experiments  │                    │
│   (Member 3)     │                    ▼
│                  │     ┌──────────────────────────────────────┐
│  Datasets        │     │       DASHBOARD + OUTPUT (Member 5)  │
│  Models          │     │  Streamlit · LaTeX Draft · Citations │
│  Metrics         │     └──────────────────────────────────────┘
│  Stats Analysis  │
└─────────────────┘
```

---

## 👥 Team Roles & Responsibilities

### 👩‍💻 Member 1 — Agent / LLM Architect
> **Builds the brain and orchestration layer**

- Designs and wires together all AI agents (Literature, Hypothesis, Code, Critic, Synthesis)
- Manages agent communication, tool calling, workflow and state
- **Tech**: LangChain / AutoGen, LangGraph, prompt engineering, structured outputs, agent memory

---

### 👩‍💻 Member 2 — Data + RAG Engineer *(this branch)*
> **Owns the scientific knowledge base**

- Fetches papers from **arXiv** and **PubMed** APIs
- Processes PDFs (text extraction, figures)
- Cleans, chunks, and embeds text
- Stores and retrieves from a **vector database**
- Provides RAG context to the Literature Agent

**Pipeline:**
```
arXiv / PubMed APIs
        ↓
   Data Ingestion
        ↓
  PDF Processing  (PyMuPDF)
        ↓
     Chunking
        ↓
    Embeddings
        ↓
    Vector DB       (ChromaDB / Milvus / Pinecone)
        ↓
    Retrieval
        ↓
      RAG
```

---

### 👩‍💻 Member 3 — ML + Experimentation + Critic
> **The most ML-heavy role — runs experiments and critiques results**

- Handles datasets, preprocessing, ML experiments
- Computes metrics, p-values, confidence intervals
- Powers the **Critic Agent** that evaluates hypotheses statistically
- **Tech**: NumPy, Pandas, scikit-learn, Matplotlib, hypothesis testing

**Pipeline:**
```
Hypothesis → Dataset → Experiment → Model → Metrics → Statistical Analysis → Critic
```

---

### 👩‍💻 Member 4 — Code Execution + Security
> **Handles safe execution of AI-generated Python code**

- Builds the sandboxed Docker execution environment
- Manages resource limits, timeouts, and file restrictions
- Returns stdout/stderr back to the agent loop
- **Tech**: Docker, Dockerfiles, subprocess, Linux, security hardening

**Pipeline:**
```
Generated Python → Safety Checks → Docker Container → Execute → stdout/stderr → Agent
```

---

### 👩‍💻 Member 5 — Multimodal + Dashboard + Research Synthesis
> **Owns the human-facing layer and multimodal understanding**

- Extracts figures/charts from PDFs using a Vision Language Model (VLM)
- Synthesises all agent results into a **LaTeX research draft**
- Builds the **Streamlit dashboard** showing agent actions, citations, costs, results
- **Tech**: Streamlit, PyMuPDF, LLaVA (or similar VLM), LLM APIs, LaTeX

**Pipeline:**
```
PDF → Figures/Charts → Vision Model → Extract Information
Results → LLM → LaTeX Research Draft
Streamlit Dashboard → Research Progress · Agent Actions · Citations · Experiments · Costs · Results
```

---

## 📁 Project Structure

```
Autonomous-Multimodal-Scientific-Discovery-Agent/
│
├── agents/                    # Member 1 — Orchestration & agent definitions
│   ├── literature_agent.py
│   ├── hypothesis_agent.py
│   ├── code_agent.py
│   ├── critic_agent.py
│   └── synthesis_agent.py
│
├── data/                      # Member 2 — Paper fetchers
│   ├── arxiv_fetcher.py
│   └── pubmed_fetcher.py
│
├── rag/                       # Member 2 — RAG pipeline
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   └── vector_store.py
│
├── experiments/               # Member 3 — ML experiments & critic
│   ├── dataset_loader.py
│   ├── trainer.py
│   ├── metrics.py
│   └── critic.py
│
├── sandbox/                   # Member 4 — Docker code execution
│   ├── executor.py
│   ├── safety_checks.py
│   └── Dockerfile
│
├── dashboard/                 # Member 5 — Streamlit UI & synthesis
│   ├── app.py
│   ├── multimodal.py
│   └── latex_writer.py
│
├── pipeline.py                # End-to-end RAG prototype (Member 2)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template (copy → .env)
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Raji1674/Autonomous-Multimodal-Scientific-Discovery-Agent.git
cd Autonomous-Multimodal-Scientific-Discovery-Agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
# Then open .env and fill in your ENTREZ_EMAIL (required for PubMed)
```

---

## 🚀 Quick Start — RAG Pipeline (Member 2)

```python
from pipeline import run_pipeline

# Fetch papers on a topic, embed them, and retrieve the most relevant chunks
results = run_pipeline(query="CRISPR gene editing cancer therapy", top_k=5)
for r in results:
    print(r["title"], "—", r["source"])
```

---

## 🧱 Technology Stack

| Layer | Component | Library / Tool |
|-------|-----------|----------------|
| Orchestration | Agent framework | LangChain / AutoGen |
| Orchestration | Workflow | LangGraph |
| RAG | arXiv API | `arxiv` |
| RAG | PubMed API | `biopython` (Entrez) |
| RAG | PDF processing | `PyMuPDF` (fitz) |
| RAG | Text cleaning | `beautifulsoup4` |
| RAG | Chunking | `langchain-text-splitters` |
| RAG | Embeddings | `sentence-transformers` |
| RAG | Vector DB | `chromadb` (local) / Milvus / Pinecone |
| ML | Experiments | NumPy, Pandas, scikit-learn |
| ML | Visualisation | Matplotlib |
| Sandbox | Code execution | Docker, subprocess |
| Dashboard | UI | Streamlit |
| Dashboard | Multimodal | LLaVA (VLM) |
| Dashboard | Report | LaTeX |

---

## 🌿 Git Branch Convention

| Member | Branch |
|--------|--------|
| Member 1 | `feature/member1-orchestration` |
| Member 2 | `feature/member2-data-rag` |
| Member 3 | `feature/member3-ml-critic` |
| Member 4 | `feature/member4-sandbox` |
| Member 5 | `feature/member5-dashboard` |

> All feature branches merge into `main` via Pull Requests.