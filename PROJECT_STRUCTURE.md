# 📂 Project Structure

This document acts as an architectural map, outlining the core responsibilities of each file within the RAG Chatbot's environment.

---

## 💾 Database Layer (`sql/`)

- **`docker-compose.yml`** — Defines the `pgvector:pg16` Docker container runtime and local data volumes.
- **`sql/schema.sql`** — Represents the original base vector layout and HNSW index assignments.
- **`sql/migrations.sql`** — Implements runtime extensions for Production. Adds TSVECTOR columns for hybrid keyword search pipelines, document size mappings, and the dynamic connection tables for conversation history persistence.

---

## 🧠 Backend Engine (`backend/`)

- **`main.py`** — The primary FastAPI orchestrator. Controls HTTP routing bindings, parallel chunk embedding logic (via Async Semaphores), API validation pipelines, and direct SSE Stream interception logic.
- **`config.py`** — Strongly typed `pydantic-settings` schema that inherently loads and protects memory state from `.env`.
- **`db.py`** — Initializes the persistent `psycopg2` PostgreSQL connection pool routing bindings.
- **`models.py`** — Provides strict JSON structure specifications for all API ingress and egress layers.
- **`chunking.py`** — Holds the text manipulation logic. It parses sentence-boundaries specifically and leverages heuristic Regex mapping to rip Header Titles implicitly off nested PDF documentation formats.
- **`embeddings.py`** — The remote HTTP binding client for talking into `nomic-embed-text` to generate spatial math logic.
- **`llm_adapter.py`** — The polymorphic proxy system interfacing seamlessly with Anthropic, OpenAI, LM Studio, or Ollama, directly formatting internal LLM inference output natively to Streamlit Server-Sent Events.
- **`parsers.py`** — File byte ingestion system extracting string literals efficiently mapped by `.docx`, `.pdf`, `.md`, etc.
- **`prompts.py`** — Injects Context arrays implicitly and bounds Multi-Turn Session history logic accurately natively off `psycopg2` array bounds.
- **`retriever.py`** — Contains the **Reciprocal Rank Fusion** module. It effectively merges pgvector cosine similarities (`<=>`) concurrently beside standard `tsquery` full text pattern evaluations into a singular scoring output.

---

## 🎨 Frontend Client (`frontend/`)

- **`app.py`** — The pure Streamlit rendering interface. Manages state layout logic natively abstracting raw byte streams safely into dynamic UI bubbles alongside raw search debugging readouts.

---

## ⚙️ CI / CD & Evaluation

- **`eval/run_eval.py`** — The diagnostic parsing loop that synthetically checks how competent local LLMs are at cross-referencing information strings correctly.
- **`.github/workflows/lint.yml`** — Automated cloud test harness acting as a gatekeeper enforcing `ruff` semantics checks against the codebase during network pushes.
