CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    source_filename TEXT NOT NULL,
    section_title TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_session
    ON documents (session_id);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING hnsw (embedding vector_cosine_ops);

-- optional: lets a "clear session" button work cleanly
CREATE INDEX IF NOT EXISTS idx_documents_created
    ON documents (created_at);