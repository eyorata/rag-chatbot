import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from db import get_conn, put_conn
from embeddings import embed
from parsers import parse_file
from chunking import chunk_text
from retriever import retrieve
from prompts import build_messages
from llm_adapter import LLMClient
from models import ChatRequest, ChatResponse, UploadResponse, SourceChunk

app = FastAPI(title="RAG Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = LLMClient()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload(session_id: str, files: list[UploadFile] = File(...)):
    total_chunks = 0
    filenames = []
    conn = get_conn()
    try:
        cur = conn.cursor()
        for file in files:
            raw = await file.read()
            text = parse_file(file.filename, raw)
            chunks = chunk_text(text, file.filename)
            for chunk in chunks:
                vector = await embed(chunk["content"])
                cur.execute(
                    """INSERT INTO documents (session_id, source_filename, section_title, content, embedding)
                       VALUES (%s, %s, %s, %s, %s::vector)""",
                    (session_id, file.filename, chunk["title"], chunk["content"], vector),
                )
                total_chunks += 1
            filenames.append(file.filename)
        conn.commit()
    finally:
        put_conn(conn)

    return UploadResponse(session_id=session_id, files=filenames, chunks_indexed=total_chunks)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    chunks = await retrieve(req.session_id, req.query)
    messages = build_messages(req.query, chunks)
    answer = await llm.chat(messages)
    return ChatResponse(
        answer=answer,
        sources=[SourceChunk(**c) for c in chunks],
    )

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE session_id = %s", (session_id,))
        conn.commit()
        return {"deleted_rows": cur.rowcount}
    finally:
        put_conn(conn)