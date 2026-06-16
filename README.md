# 💬 RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you chat with your own documents. Upload PDFs, Word files, or Markdown, and ask questions — the system retrieves the most relevant passages and sends them to a local LLM (via Ollama or LM Studio) to generate a grounded answer.

---

## ✨ Features

- 📄 **Multi-format document ingestion** — PDF, DOCX, TXT, Markdown
- 🔍 **Semantic search** using vector embeddings (pgvector + cosine similarity)
- 🤖 **Flexible LLM backend** — Ollama, LM Studio, OpenAI, or Anthropic
- 🧠 **Session-based context** — each browser session has its own isolated document store
- 🗂️ **Chunking with overlap** — sliding-window chunking with section-aware Markdown splitting
- 📊 **Source attribution** — every answer shows which file and section it came from
- 🧪 **Evaluation harness** — keyword-hit eval runner included

---

## 🏗️ Architecture

```
┌─────────────────┐        ┌──────────────────┐        ┌─────────────────┐
│   Streamlit UI  │◄──────►│  FastAPI Backend  │◄──────►│  pgvector (DB)  │
│  (frontend/)    │        │   (backend/)      │        │  Docker         │
└─────────────────┘        └────────┬─────────┘        └─────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   Ollama / LM Studio│
                          │  (embeddings + LLM) │
                          └─────────────────────┘
```

**Upload flow:**  
File → parse → chunk → embed (Ollama) → store in pgvector

**Chat flow:**  
Query → embed → cosine similarity search → top-k chunks → LLM prompt → answer

---

## 📁 Project Structure

```
rag-chatbot/
├── docker-compose.yml       # pgvector database
├── .env.example             # environment variable template
├── sql/
│   └── schema.sql           # table + HNSW index definitions
├── backend/
│   ├── main.py              # FastAPI app & endpoints
│   ├── config.py            # pydantic-settings config
│   ├── db.py                # psycopg2 connection pool
│   ├── embeddings.py        # Ollama embedding client
│   ├── llm_adapter.py       # LLM client (Ollama/OpenAI/Anthropic)
│   ├── retriever.py         # vector similarity retrieval
│   ├── prompts.py           # system prompt + message builder
│   ├── parsers.py           # PDF / DOCX / TXT / MD parsers
│   ├── chunking.py          # sliding-window + markdown chunking
│   ├── models.py            # Pydantic request/response models
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit UI
│   └── requirements.txt
├── eval/
│   ├── eval_questions.json  # evaluation question set
│   ├── run_eval.py          # evaluation runner
│   └── results/             # JSON result files (gitignored)
└── sample_docs/
    └── sample.pdf
```

---

## ⚙️ Setup

### Prerequisites

| Service | Purpose |
|---|---|
| Docker + Docker Compose | pgvector database |
| Ollama (local or remote) | Embeddings + LLM inference |
| Python 3.11+ | Backend & frontend |

---

### 1. Start the database

```bash
docker compose up -d
```

This starts a `pgvector/pgvector:pg16` container on port **5445**.

### 2. Apply the schema

```bash
docker exec -i rag-chatbot-db-1 psql -U rag -d ragdb < sql/schema.sql
```

### 3. Configure the backend

```bash
cp .env.example backend/.env
# Edit backend/.env with your values
```

**`backend/.env` reference:**

```env
DATABASE_URL=postgresql://rag:ragpass@localhost:5445/ragdb

# Ollama (local or remote)
EMBEDDING_BASE_URL=http://localhost:11434
OLLAMA_BASE_URL=http://localhost:11434

# Embedding model (must be pulled in Ollama)
EMBEDDING_MODEL=nomic-embed-text

# LLM settings
LLM_PROVIDER=ollama        # ollama | lmstudio | openai | anthropic
LLM_MODEL=qwen3:latest

# Optional (for OpenAI / Anthropic)
LLM_API_KEY=

# Retrieval settings
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=0.3
```

### 4. Pull required Ollama models

```bash
ollama pull nomic-embed-text   # embedding model
ollama pull qwen3:latest       # or whichever LLM you set
```

### 5. Install dependencies

```bash
# Backend
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Frontend (separate terminal)
cd frontend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 6. Run the backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

### 7. Run the frontend

```bash
cd frontend
source venv/bin/activate
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🚀 Usage

1. Open the Streamlit UI at `http://localhost:8501`
2. Upload one or more documents (PDF, DOCX, TXT, MD) in the sidebar
3. Click **Index files** — chunks are embedded and stored in pgvector
4. Type a question in the chat input
5. The answer appears with source citations and similarity scores
6. Click **Clear session documents** to reset

---

## 🧪 Evaluation

Edit `eval/eval_questions.json` with your questions and expected keywords, then:

```bash
# Make sure the backend is running and documents are indexed
python eval/run_eval.py my_model_name
# Results saved to eval/results/my_model_name.json
```

---

## 🔧 Supported LLM Providers

| Provider | `LLM_PROVIDER` value | Notes |
|---|---|---|
| Ollama | `ollama` | Default. Local or remote. |
| LM Studio | `lmstudio` | OpenAI-compatible API |
| OpenAI | `openai` | Set `LLM_API_KEY` |
| Anthropic | `anthropic` | Set `LLM_API_KEY` |

---

## 📝 Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Ollama host for embeddings |
| `LLM_PROVIDER` | `ollama` | LLM backend |
| `LLM_MODEL` | `qwen3.5:latest` | Model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama host for LLM |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234` | LM Studio host |
| `LLM_API_KEY` | *(empty)* | API key for OpenAI/Anthropic |
| `RETRIEVAL_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity |

---

## 📄 License

MIT
