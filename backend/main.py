import logging
import asyncio
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from db import get_conn, put_conn
from embeddings import embed
from parsers import parse_file
from chunking import chunk_text
from retriever import retrieve, retrieve_raw
from prompts import build_messages
from llm_adapter import LLMClient
from config import settings
from models import (
    ChatRequest, ChatResponse, UploadResponse, SourceChunk, 
    DocumentInfo, HistoryMessage, SearchResponse, ModelList
)

# Configure structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = LLMClient()


# ─────────────────────────────────────────────────────────────────────────────
# Startup Validation
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    try:
        # Test DB
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        put_conn(conn)
        logger.info("Database connection OK")
    except Exception as e:
        logger.critical(f"FATAL: Database connection failed on startup: {e}")
        import sys
        sys.exit(1)

    try:
        # Test Embeddings API
        res = await embed("test")
        if res is None:
            raise ValueError("Embedding API returned None")
        logger.info("Embedding API connection OK")
    except Exception as e:
        logger.critical(f"FATAL: Embedding API connection failed on startup: {e}")
        import sys
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_ok = False
    try:
        conn = get_conn()
        conn.cursor().execute("SELECT 1")
        put_conn(conn)
        db_ok = True
    except:
        pass
        
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except:
        pass

    return {
        "status": "ok" if db_ok and ollama_ok else "degraded",
        "db": db_ok,
        "ollama": ollama_ok
    }


@app.get("/models", response_model=ModelList)
async def list_models():
    """Fetch available models dynamically from Ollama."""
    if settings.llm_provider != "ollama":
        return ModelList(llm_models=[settings.llm_model], embed_models=[settings.embedding_model])
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            r.raise_for_status()
            
            all_models = [m["name"] for m in r.json().get("models", [])]
            embed_models = [m for m in all_models if "embed" in m.lower()]
            llm_models = [m for m in all_models if "embed" not in m.lower()]
            
            return ModelList(llm_models=llm_models, embed_models=embed_models)
    except Exception as e:
        logger.warning(f"Could not fetch models from Ollama: {e}")
        return ModelList(llm_models=[settings.llm_model], embed_models=[settings.embedding_model])


@app.post("/upload", response_model=UploadResponse)
async def upload(session_id: str, files: list[UploadFile] = File(...)):
    chunks_indexed = 0
    chunks_failed = 0
    files_processed = []
    files_failed = []
    errors = []
    
    conn = get_conn()
    try:
        cur = conn.cursor()
        for file in files:
            # 1. Size Validation
            file_raw = await file.read()
            size_mb = len(file_raw) / (1024 * 1024)
            if size_mb > settings.max_file_size_mb:
                err = f"{file.filename} exceeds {settings.max_file_size_mb}MB limit"
                logger.warning(err)
                errors.append(err)
                files_failed.append(file.filename)
                continue

            try:
                # 2. Duplicate Detection: Delete old chunks for this file + session
                cur.execute(
                    "DELETE FROM documents WHERE session_id = %s AND source_filename = %s",
                    (session_id, file.filename)
                )
                
                # 3. Parse & Chunk
                text = parse_file(file.filename, file_raw)
                chunks = chunk_text(text, file.filename)
                if not chunks:
                    logger.info(f"Skipping empty file {file.filename}")
                    continue

                # 4. Concurrent Embedding with Semaphore limit
                sem = asyncio.Semaphore(settings.embed_concurrency)
                
                async def embed_chunk(c):
                    async with sem:
                        vector = await embed(c["content"])
                        return c, vector

                tasks = [embed_chunk(c) for c in chunks]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 5. Insert to DB
                for res in results:
                    if isinstance(res, Exception) or res[1] is None:
                        chunks_failed += 1
                        continue
                        
                    chunk_data, vector = res
                    cur.execute(
                        """INSERT INTO documents (session_id, source_filename, section_title, content, embedding, file_size_bytes)
                           VALUES (%s, %s, %s, %s, %s::vector, %s)""",
                        (session_id, file.filename, chunk_data["title"], chunk_data["content"], vector, len(file_raw)),
                    )
                    chunks_indexed += 1
                
                files_processed.append(file.filename)
                logger.info(f"Indexed {file.filename} for session {session_id}")
                
            except Exception as e:
                err = f"Failed to process {file.filename}: {str(e)}"
                logger.error(err)
                errors.append(err)
                files_failed.append(file.filename)
                
        conn.commit()
    finally:
        put_conn(conn)

    return UploadResponse(
        session_id=session_id,
        files_processed=files_processed,
        files_failed=files_failed,
        chunks_indexed=chunks_indexed,
        chunks_failed=chunks_failed,
        errors=errors
    )


def _get_history(cur, session_id: str) -> list[dict]:
    """Fetch last N turns of conversation history."""
    cur.execute(
        """SELECT role, content FROM conversation_history 
           WHERE session_id = %s 
           ORDER BY created_at DESC LIMIT %s""",
        (session_id, settings.history_turns * 2) # * 2 because user+assistant = 2 rows/turn
    )
    rows = cur.fetchall()
    # Rows are returned newest-first, we need oldest-first for the prompt
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def _get_available_docs(cur, session_id: str) -> list[str]:
    cur.execute("SELECT DISTINCT source_filename FROM documents WHERE session_id = %s", (session_id,))
    return [r[0] for r in cur.fetchall()]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        chunks = await retrieve(req.session_id, req.query, settings.llm_model)
        
        conn = get_conn()
        try:
            cur = conn.cursor()
            history = _get_history(cur, req.session_id)
            docs = _get_available_docs(cur, req.session_id)
            
            # Save User query to history
            cur.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (req.session_id, "user", req.query)
            )
        finally:
            put_conn(conn)
            
        messages = build_messages(req.query, chunks, history, docs)
        logger.info(f"Chat request - session: {req.session_id}, query: '{req.query}', using {len(chunks)} chunks, {len(history)} history msgs")
        
        try:
            answer = await llm.chat(messages)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM API Error: {str(e)}")

        conn = get_conn()
        try:
            cur = conn.cursor()
            # Save Assistant answer to history
            cur.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (req.session_id, "assistant", answer)
            )
            conn.commit()
        finally:
            put_conn(conn)
            
        top_sim = chunks[0]["similarity"] if chunks else 0.0

        return ChatResponse(
            answer=answer,
            sources=[SourceChunk(**c) for c in chunks],
            chunks_retrieved=len(chunks),
            top_similarity=top_sim
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during chat")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming version of /chat using SSE."""
    try:
        chunks = await retrieve(req.session_id, req.query, settings.llm_model)
        
        conn = get_conn()
        try:
            cur = conn.cursor()
            history = _get_history(cur, req.session_id)
            docs = _get_available_docs(cur, req.session_id)
            
            # Save User query to history
            cur.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (req.session_id, "user", req.query)
            )
            conn.commit()
        finally:
            put_conn(conn)
            
        messages = build_messages(req.query, chunks, history, docs)
        
        # We need to capture the full answer as it streams to save it to DB later.
        # Since we yield directly, we wrap the generator to intercept tokens.
        async def intercept_stream():
            full_answer = []
            import json
            async for sse_chunk in llm.chat_stream(messages):
                yield sse_chunk
                try:
                    # extract 'content' from SSE JSON data to construct the full string
                    if sse_chunk.startswith("data: "):
                        data = json.loads(sse_chunk[6:])
                        if "content" in data:
                            full_answer.append(data["content"])
                except:
                    pass
            
            # Streaming done, save answer to DB
            final_text = "".join(full_answer)
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                    (req.session_id, "assistant", final_text)
                )
                # Also yield the sources as a final event so the UI can display them
                top_sim = chunks[0]["similarity"] if chunks else 0.0
                stats = {
                    "sources": chunks,
                    "chunks_retrieved": len(chunks),
                    "top_similarity": top_sim
                }
                yield f"data: {json.dumps({'metadata': stats})}\n\n"
                conn.commit()
            except Exception as e:
                logger.error(f"Error saving history after stream: {e}")
            finally:
                put_conn(conn)

        return StreamingResponse(intercept_stream(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during chat stream")


@app.get("/search", response_model=SearchResponse)
async def search(session_id: str, query: str):
    """Raw retrieval without LLM processing — useful for debugging keyword vs vector matches."""
    chunks = await retrieve_raw(session_id, query)
    return SearchResponse(chunks=[SourceChunk(**c) for c in chunks])


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """"Return full conversation history for a session."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, created_at FROM conversation_history WHERE session_id = %s ORDER BY created_at ASC",
            (session_id,)
        )
        history = [HistoryMessage(role=r[0], content=r[1], timestamp=r[2].isoformat()) for r in cur.fetchall()]
        return {"session_id": session_id, "history": history}
    finally:
        put_conn(conn)


@app.get("/documents/{session_id}")
async def get_documents(session_id: str):
    """List all mapped documents in a session."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT source_filename, count(*) as chunk_count, min(created_at) as uploaded, max(file_size_bytes) as size
               FROM documents WHERE session_id = %s GROUP BY source_filename""",
            (session_id,)
        )
        docs = [
            DocumentInfo(
                filename=r[0], 
                chunk_count=r[1], 
                upload_timestamp=r[2].isoformat(),
                file_size_bytes=r[3]
            ) for r in cur.fetchall()
        ]
        return {"session_id": session_id, "documents": docs}
    finally:
        put_conn(conn)


@app.delete("/documents/{session_id}/{filename}")
async def delete_document(session_id: str, filename: str):
    """Delete a specific document while preserving the rest of the session."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE session_id = %s AND source_filename = %s", (session_id, filename))
        conn.commit()
        return {"deleted_rows": cur.rowcount, "filename": filename}
    finally:
        put_conn(conn)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear all documents and conversation history for a session."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE session_id = %s", (session_id,))
        doc_rows = cur.rowcount
        cur.execute("DELETE FROM conversation_history WHERE session_id = %s", (session_id,))
        hist_rows = cur.rowcount
        conn.commit()
        logger.info(f"Cleared session {session_id} - docs deleted: {doc_rows}, history deleted: {hist_rows}")
        return {"deleted_document_chunks": doc_rows, "deleted_history_turns": hist_rows}
    finally:
        put_conn(conn)