# 🔬 Autonomous Multimodal Scientific Discovery Agent

An AI-powered research assistant that autonomously retrieves scientific literature,
designs and runs ML experiments, critiques results statistically, and synthesises
findings into a human-readable research report — all driven by a multi-agent LLM pipeline.

---

## 🏗️ Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER  (Member 1)                  │
│          LangChain / AutoGen  ·  LangGraph  ·  State Management      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────┐   ┌──────────────────────────────────────────────┐
│  Knowledge Base  │──▶│               AGENT PIPELINE                 │
│   (Member 2)     │   │                                              │
│                  │   │   Literature Agent                           │
│  arXiv / PubMed  │   │         ↓                                    │
│  PDF Processing  │   │   Hypothesis Agent                           │
│  Chunking        │   │         ↓                                    │
│  Embeddings      │   │   Code Agent ──────► Sandbox  (Member 4)    │
│  Vector DB       │   │         ↓                                    │
│  RAG Retrieval   │   │   Critic Agent              (Member 3)       │
└─────────────────┘   │         ↓                                    │
                       │   Synthesis Agent            (Member 5)       │
┌─────────────────┐   └──────────────────────────────────────────────┘
│  ML Experiments  │                      │
│   (Member 3)     │                      ▼
│                  │   ┌──────────────────────────────────────────────┐
│  Datasets        │   │        DASHBOARD + OUTPUT  (Member 5)         │
│  Models          │   │  Streamlit · LaTeX Draft · Citations          │
│  Metrics         │   └──────────────────────────────────────────────┘
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
- **Branch**: `feature/member1-orchestration`

---

### 👩‍💻 Member 2 — Data + RAG Engineer
> **Owns the scientific knowledge base**

- Fetches papers from **arXiv** and **PubMed** APIs
- Processes PDFs (text extraction, figures) using PyMuPDF
- Cleans, chunks, and embeds text into a vector database
- Provides semantic RAG retrieval to the Literature Agent
- **Tech**: `arxiv`, `biopython`, `PyMuPDF`, `sentence-transformers`, `chromadb`, `langchain-text-splitters`
- **Branch**: `feature/member2-data-rag`

**Pipeline:**
```
arXiv / PubMed APIs  →  Data Ingestion  →  PDF Processing
    →  Chunking  →  Embeddings  →  Vector DB  →  RAG Retrieval
```

---

### 👩‍💻 Member 3 — ML + Experimentation + Critic
> **The most ML-heavy role — runs experiments and critiques results statistically**

- Handles datasets, preprocessing, ML experiments and model training
- Computes metrics, p-values, and confidence intervals
- Powers the **Critic Agent** that evaluates whether results are statistically valid
- **Tech**: NumPy, Pandas, scikit-learn, Matplotlib, hypothesis testing
- **Branch**: `feature/member3-ml-critic`

**Pipeline:**
```
Hypothesis  →  Dataset  →  Experiment  →  Model
    →  Metrics  →  Statistical Analysis  →  Critic
```

---

### 👩‍💻 Member 4 — Code Execution + Security
> **Handles safe execution of AI-generated Python code inside Docker**

- Builds the sandboxed Docker execution environment
- Manages resource limits, timeouts, and file restrictions
- Returns `stdout` / `stderr` back to the agent loop
- **Tech**: Docker, Dockerfiles, subprocess, Linux, security hardening
- **Branch**: `feature/member4-sandbox`

**Pipeline:**
```
Generated Python  →  Safety Checks  →  Docker Container
    →  Execute  →  stdout / stderr  →  Agent
```

---

### 👩‍💻 Member 5 — Multimodal + Dashboard + Research Synthesis
> **Owns the human-facing layer, multimodal understanding, and final report**

- Extracts figures and charts from PDFs using a Vision Language Model (VLM)
- Synthesises all agent results into a **LaTeX research draft**
- Builds the **Streamlit dashboard** showing agent actions, citations, costs, and results
- **Tech**: Streamlit, PyMuPDF, LLaVA (or similar VLM), LLM APIs, LaTeX
- **Branch**: `feature/member5-dashboard`

**Pipeline:**
```
PDF  →  Figures / Charts  →  Vision Model  →  Extract Information
Results  →  LLM  →  LaTeX Research Draft
Streamlit Dashboard  →  Progress · Actions · Citations · Experiments · Costs · Results
```

---

## 📁 Full Project Structure

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
│   ├── arxiv_fetcher.py       # ✅ Done
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
├── pipeline.py                # End-to-end RAG demo (Member 2)
├── requirements.txt           # All Python dependencies
├── .env.example               # Environment variable template — copy to .env
└── README.md                  # This file
```

---

## ⚙️ Setup (All Members)

### 1. Clone the repo
```bash
git clone https://github.com/Raji1674/Autonomous-Multimodal-Scientific-Discovery-Agent.git
cd Autonomous-Multimodal-Scientific-Discovery-Agent
```

### 2. Checkout your branch
```bash
git checkout feature/member2-data-rag   # change to your own branch
```

### 3. Create a virtual environment
```bash
python -m venv venv

# Activate — Windows
.\venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure environment variables
```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
# Open .env and fill in ENTREZ_EMAIL (required for PubMed)
```

---

## 🌿 Git Branch Convention

| Member | Role | Branch |
|--------|------|--------|
| Member 1 | Agent / LLM Architect | `feature/member1-orchestration` |
| Member 2 | Data + RAG Engineer | `feature/member2-data-rag` |
| Member 3 | ML + Experimentation + Critic | `feature/member3-ml-critic` |
| Member 4 | Code Execution + Security | `feature/member4-sandbox` |
| Member 5 | Multimodal + Dashboard + Synthesis | `feature/member5-dashboard` |

> All feature branches merge into `main` via Pull Requests. Never push directly to `main`.

---

## 🧱 Full Technology Stack

| Layer | Component | Library / Tool |
|-------|-----------|----------------|
| Orchestration | Agent framework | LangChain / AutoGen |
| Orchestration | Workflow & state | LangGraph |
| RAG | arXiv API | `arxiv` |
| RAG | PubMed API | `biopython` (Entrez) |
| RAG | PDF processing | `PyMuPDF` (fitz) |
| RAG | Text cleaning | `beautifulsoup4` |
| RAG | Chunking | `langchain-text-splitters` |
| RAG | Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| RAG | Vector DB | `chromadb` (local) |
| ML | Experiments | NumPy, Pandas, scikit-learn |
| ML | Visualisation | Matplotlib |
| Sandbox | Code execution | Docker, subprocess |
| Dashboard | UI | Streamlit |
| Dashboard | Multimodal | LLaVA (VLM) |
| Dashboard | Report | LaTeX |