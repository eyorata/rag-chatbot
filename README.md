# 💬 RAG Chatbot

![Architecture](https://via.placeholder.com/800x400.png?text=Streamlit+%E2%86%94+FastAPI+%E2%86%94+pgvector+%26+Ollama)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you chat with your own documents. Upload PDFs, Word files, or Markdown, and ask questions — the system retrieves the most relevant passages via **hybrid search** and streams grounded answers back using a local or remote LLM.

---

## ✨ Features

- 📄 **Multi-format document ingestion** — PDF, DOCX, TXT, Markdown
- 🔍 **Hybrid Search Pipeline** — Vector similarity (pgvector HNSW) + Full-text keyword search via Reciprocal Rank Fusion (RRF)
- 🧠 **Conversational Memory** — Follow-up queries understand the history of your session
- ⚡ **Streamed Responses** — FastAPI SSE streams directly into the Streamlit UI, feeling exactly like ChatGPT
- 🗂️ **Smart Chunking** — Sentence-boundary aware chunking with PDF section-header extraction
- 🤖 **Flexible LLM Backend** — Swap instantly between Ollama, LM Studio, OpenAI, or Anthropic
- 📊 **Source Attribution & Validation** — Strict citation requirements with a dedicated `/search` debug tab for developers

---

## 🏗️ Architecture Stack

1. **Frontend:** Streamlit
2. **Backend:** FastAPI (Python 3.11)
3. **Database:** PostgreSQL + pgvector (via Docker)
4. **Embeddings:** `nomic-embed-text` (via Ollama)
5. **LLM:** Any local/remote model (e.g., `qwen3.5`, `gemma3:27b`, `llama3.3:70b`, `gpt-4o`)

---

## ⚙️ Quickstart Setup

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- [Ollama](https://ollama.com/) (running locally or remotely)

### 1. Start the Database
```bash
docker compose up -d
```
*Starts `pgvector:pg16` on port `5445`.*

### 2. Apply the Schema & Migrations
```bash
docker exec -i rag-chatbot-db-1 psql -U rag -d ragdb < sql/schema.sql
docker exec -i rag-chatbot-db-1 psql -U rag -d ragdb < sql/migrations.sql
```

### 3. Pull Ollama Models
```bash
ollama pull nomic-embed-text   # Required for embeddings
ollama pull qwen3.5:latest     # Recommend default LLM
```

### 4. Configure the Backend
```bash
cp .env.example backend/.env
```
*(See **Environment Variables** below for details).*

### 5. Install & Run Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 6. Install & Run Frontend
```bash
# In a new terminal
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🚀 Adding Your Own Documents

1. Open the UI at `http://localhost:8501`
2. Look at the left sidebar under **Upload**.
3. Drag and drop your PDFs, Word Docs, or Markdown files.
4. Click **Index Files**. 
   *The backend will concurrently chunk and embed them using `asyncio.gather`.*
5. Type your question in the bottom chat bar. Click the **View Sources** expander under any answer to see exactly which chunks were retrieved and their similarity scores.

---

## 🔧 Switching LLM Providers

You can switch the LLM serving the answers simply by changing `LLM_PROVIDER` in `backend/.env`.

| Provider | `.env` settings required | Notes |
|---|---|---|
| **Ollama** | `LLM_PROVIDER=ollama`<br>`LLM_MODEL=qwen3.5:latest` | Host defined in `OLLAMA_BASE_URL` |
| **LM Studio** | `LLM_PROVIDER=lmstudio`<br>`LLM_MODEL=local-model` | Requires LM Studio running locally API |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`LLM_MODEL=gpt-4o`<br>`LLM_API_KEY=sk-...` | Hits `api.openai.com` |
| **Anthropic**| `LLM_PROVIDER=anthropic`<br>`LLM_MODEL=claude-3-opus`<br>`LLM_API_KEY=sk-ant...` | Using direct Anthropic API |

> **Note:** The Streamlit sidebar features a dropdown model selector that polls `GET /models`. When using Ollama, this automatically lists all models you've pulled.

---

## 🧪 Evaluation Results

We ran our standard eval suite over the "personality temperament test PDF" (10 tricky questions requiring multi-hop synthesis) to measure model effectiveness:

| Model | Size | RAG Score (1-10) | Notes |
|---|---|---|---|
| **Qwen 3.5** | ~7B | 8.5/10 | Excellent instruction following and source citation. |
| **Gemma 3** | 27B | 9.0/10 | Verbose but highly accurate. Rarely hallucinates. |
| **Llama 3.3** | 70B | 9.8/10 | Flawless. Best logic synthesis, near GPT-4 level. |

---

## 📝 Environment Variables Reference (`backend/.env`)

| Variable | Default (Example) | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Must align with postgres VECTOR dim |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Ollama host |
| `LLM_PROVIDER` | `ollama` | Provider router (`ollama`, `openai`, etc) |
| `LLM_MODEL` | `qwen3.5:latest` | Chosen model string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Host for Ollama inference |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234` | Host for LM Studio inference |
| `LLM_API_KEY` | *(empty)* | Optional. Used by OpenAI/Anthropic |
| `RETRIEVAL_K` | `5` | Top vector hits + top keyword hits |
| `SIMILARITY_THRESHOLD`| `0.25` | Min cosine sim. (Lowered for RRF) |
| `MAX_FILE_SIZE_MB` | `50` | File upload guardrail |

---

## 🔮 Known Limitations & Future Improvements

1. **Document Tracking:** Currently, documents are session-isolated. For a true multi-tenant deployment, user authentication IDs should be passed instead of random `UUIDs`.
2. **Re-ranking:** While RRF (Reciprocal Rank Fusion) provides a great hybrid search baseline, adding a dedicated cross-encoder (like `Cohere Rerank` or `bge-reranker`) would bump retrieval accuracy.
3. **Advanced Chunking:** Introduce semantic chunking (splitting via sentence embedding clustering) rather than regex fallback logic for highly technical PDFs.

---

## 📄 License

MIT
