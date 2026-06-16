-- migrations.sql
-- Apply AFTER schema.sql. Safe to re-run (uses IF NOT EXISTS / IF column missing guards).
-- Run: docker exec -i rag-chatbot-db-1 psql -U rag -d ragdb < sql/migrations.sql

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Hybrid search: add full-text search column + GIN index to documents
-- ─────────────────────────────────────────────────────────────────────────────

-- Generated tsvector enables fast full-text keyword search alongside vector similarity.
-- Using STORED so the column is pre-computed at insert time (no query-time overhead).
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_documents_tsv
    ON documents USING GIN (content_tsv);

-- File size stored so the /documents/{session_id} endpoint can report it without
-- keeping a separate metadata table. Nullable because old rows won't have it.
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Conversation history: persists chat turns so the LLM gets memory
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversation_history (
    id         SERIAL PRIMARY KEY,
    session_id UUID    NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- Fast lookup of the last N turns for a session
CREATE INDEX IF NOT EXISTS idx_conv_session_time
    ON conversation_history (session_id, created_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Query logs: stores per-request retrieval stats for offline analysis
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS query_logs (
    id                 SERIAL PRIMARY KEY,
    session_id         UUID    NOT NULL,
    query              TEXT    NOT NULL,
    num_chunks         INT     NOT NULL DEFAULT 0,
    top_similarity     FLOAT,
    -- Track which model answered so you can compare providers in the eval harness
    llm_model          TEXT,
    created_at         TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_session
    ON query_logs (session_id, created_at DESC);
