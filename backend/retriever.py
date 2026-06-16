from db import get_conn, put_conn
from embeddings import embed
from config import settings

async def retrieve(session_id: str, query: str) -> list[dict]:
    query_embedding = await embed(query)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_filename, section_title, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE session_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, session_id, query_embedding, settings.retrieval_k),
        )
        rows = cur.fetchall()
    finally:
        put_conn(conn)

    return [
        {"source_filename": r[0], "section_title": r[1], "content": r[2], "similarity": r[3]}
        for r in rows
        if r[3] >= settings.similarity_threshold
    ]