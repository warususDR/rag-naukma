# NaUKMA RAG Assistant

**Multilingual Retrieval-Augmented Generation system for Ukrainian-language university documentation**

Master’s thesis project  

---

## Overview

This repository contains a full-stack RAG system designed specifically for Ukrainian-language institutional documents. The system was developed and evaluated on a corpus of **333 official NaUKMA PDF documents** (orders, regulations, and internal policies).

The project implements both a **naive vector RAG** baseline and an advanced **LightRAG**-based architecture that combines:

- Vector retrieval (PostgreSQL + pgvector)
- Knowledge-graph retrieval (Neo4j)
- Hybrid query modes (`naive`, `local`, `global`, `hybrid`)
- Local LLM inference via Ollama (MamayLM 12B)
- Multilingual embeddings (BAAI/bge-m3)

A modern web frontend provides a chat interface with conversation history and Google authentication.

---

## Key Features

- **Ukrainian-first design** — system prompts, entity types, and generation prioritize Ukrainian. The core LLM also provides English support.
- **Dual storage architecture** — PostgreSQL (pgvector) + Neo4j knowledge graph
- **Four retrieval modes** supported by LightRAG:
  - `naive` — pure vector search
  - `local` — entity-centric graph retrieval
  - `global` — community-level graph retrieval
  - `hybrid` — combination of vector + graph
- **Document processing pipeline** based on Marker for high-quality PDF text extraction
- **8 custom NER entity types** tailored to university documents:  
  Person, Organization, Location, Event, Concept, Document, Date, Artifact
- **Evaluation suite** using RAGAS and GraphRAG-style LLM-as-a-Judge metrics on a 130-question dataset
- **Docker Compose** deployment of the full stack

---

## Architecture

```
React Frontend
        |
        v
Flask Backend + LightRAG
        |
   +----+----+----+
   |         |    |
   v         v    v
PostgreSQL  Neo4j  Ollama
+ pgvector  Graph  (MamayLM 12B)
```

**Components:**

- **Frontend** — React + Vite + Tailwind (chat UI, conversation history, Google OAuth)
- **Backend** — Flask API wrapping LightRAG
- **Vector store** — PostgreSQL with pgvector
- **Graph store** — Neo4j
- **LLM** — MamayLM 12B served locally via Ollama
- **Embeddings** — BAAI/bge-m3

---

## Repository Structure

```
rag-naukma/
├── lightrag_pipeline/          # Core LightRAG backend + Flask API
│   ├── app.py                  # Main API server
│   ├── lightrag_model.py       # LightRAG wrapper & query logic
│   ├── extract_documents.py    # PDF to text pipeline
│   ├── graph_extraction.py
│   ├── collect_answers.py      # Batch answer generation for evaluation
│   └── Dockerfile
├── naive_rag/                  # Baseline naive vector RAG implementation
├── naukma-rag-frontend/        # React + Vite + Tailwind frontend
├── evaluation/                 # RAGAS + head-to-head evaluation scripts + results
│   ├── questions_batch*.json   # 130-question test set
│   ├── eval_summary.json
│   └── head_to_head_results.json
├── docker-compose.yml
└── requirements.txt
```

---

## Evaluation Results (130 questions)

### RAGAS Metrics (0–1)

| Metric               | Hybrid | Naive  | Local  |
|----------------------|--------|--------|--------|
| Faithfulness         | 0.962  | **0.975** | 0.767 |
| Answer Correctness   | 0.790  | **0.797** | 0.486 |
| Context Relevance    | **0.809** | 0.806 | 0.526 |
| Context Recall       | 0.921  | **0.945** | 0.595 |

### GraphRAG-style Head-to-Head (Hybrid vs Naive)

| Metric            | Hybrid Wins | Naive Wins | Tie |
|-------------------|-------------|------------|-----|
| Comprehensiveness | 45%         | 45%        | 10% |
| Diversity         | **65%**     | 30%        | 5%  |
| Empowerment       | **55%**     | 40%        | 5%  |

Hybrid mode shows clear advantages in diversity and empowerment while remaining competitive on faithfulness and correctness.

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Ollama running on the host with the required models:

```bash
ollama pull mamaylm:12b
ollama pull bge-m3
```

### 1. Clone and configure

```bash
git clone https://github.com/warususDR/rag-naukma.git
cd rag-naukma
cp .env.example .env
```

Required environment variables (example):

```env
POSTGRES_USER=rag
POSTGRES_PASSWORD=your_password
POSTGRES_DB=naukma_rag
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
FRONTEND_URL=http://localhost:3000
VITE_GOOGLE_CLIENT_ID=your_google_oauth_client_id
```

### 2. Launch the stack

```bash
docker compose up --build
```

| Service   | URL                      |
|-----------|--------------------------|
| Frontend  | http://localhost:3000    |
| Backend   | http://localhost:5000    |
| Neo4j     | http://localhost:7474    |
| Postgres  | localhost:5432           |

---

## Local Development

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m lightrag_pipeline.app
```

### Frontend

```bash
cd naukma-rag-frontend
npm install
npm run dev
```

---

## Document Ingestion

1. Place PDF files in a directory.
2. Run the extraction pipeline:

```bash
python -m lightrag_pipeline.extract_documents --input_dir path/to/pdfs --output documents.json
```

3. Ingest into LightRAG:

```python
from lightrag_pipeline.lightrag_model import LightRAGModel
model = LightRAGModel(...)
model.insert_from_json("documents.json")
```

---

## Evaluation

The `evaluation/` folder contains:

- A curated set of 130 questions grounded in NaUKMA documents
- Scripts for collecting answers from different modes
- RAGAS metric computation
- Head-to-head LLM-as-a-Judge comparison (Comprehensiveness, Diversity, Empowerment)

Example:

```bash
python evaluation/head_to_head.py
python evaluation/add_context_metrics.py
```

---

## Thesis Context

This work was carried out as a Master's thesis focused on:

1. Researching modern RAG architectures and their suitability for Ukrainian language data
2. Building a production-ready system with graph + vector retrieval
3. Rigorous evaluation on real university documentation using LLM-as-a-Judge paradigms

---

## Future Work

- Query routing / automatic mode selection
- Reranking of retrieved contexts
- Multimodal support (tables & images from PDFs)
- Scaling to larger LLMs and cloud deployment
- Potential integration of RDF and Agents

---

## License

This project is released for academic and research purposes.  
Please cite the thesis (https://ekmair.ukma.edu.ua/items/50c476eb-6925-4552-a19b-ca284fd6cb00) if you use this work.
