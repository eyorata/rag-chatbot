import logging
import asyncio
from db import get_conn, put_conn
from embeddings import embed
from config import settings

logger = logging.getLogger(__name__)

# RRF constant — higher k means rank differences matter less.
# k=60 is the standard value from the original Cormack et al. (2009) paper.
_RRF_K = 60


async def retrieve(session_id: str, query: str, llm_model: str = "") -> list[dict]:
    """Hybrid retrieval: combines vector similarity + BM25-style keyword search via RRF.

    Why RRF instead of score averaging? Scores from different retrieval systems are
    not comparable (cosine similarity vs. BM25-like tf-idf), so we rank-fuse instead.
    Chunks that appear in BOTH result sets get a strong boost; keyword-only or
    vector-only matches still surface but rank lower.
    """
    query_embedding = await embed(query)
    if query_embedding is None:
        logger.error("Embedding call failed for query; returning empty results")
        return []

    conn = get_conn()
    try:
        cur = conn.cursor()

        # ── Vector similarity search ──────────────────────────────────────────
        cur.execute(
            """
            SELECT id, source_filename, section_title, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE session_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, session_id, query_embedding, settings.retrieval_k * 2),
        )
        vector_rows = cur.fetchall()

        # ── Full-text keyword search ──────────────────────────────────────────
        cur.execute(
            """
            SELECT id, source_filename, section_title, content,
                   ts_rank(content_tsv, plainto_tsquery('english', %s)) AS text_score
            FROM documents
            WHERE session_id = %s
              AND content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY text_score DESC
            LIMIT %s
            """,
            (query, session_id, query, settings.retrieval_k * 2),
        )
        text_rows = cur.fetchall()

        # ── Log retrieval stats ───────────────────────────────────────────────
        top_score = vector_rows[0][4] if vector_rows else 0.0
        try:
            cur.execute(
                """
                INSERT INTO query_logs (session_id, query, num_chunks, top_similarity, llm_model)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, query, len(vector_rows), top_score, llm_model),
            )
            conn.commit()
        except Exception as log_err:
            # Log insert failure must never kill a user request
            logger.warning("Failed to write query log: %s", log_err)
            conn.rollback()

    finally:
        put_conn(conn)

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    # Build rank maps keyed by document id
    vector_rank = {row[0]: rank for rank, row in enumerate(vector_rows, start=1)}
    text_rank   = {row[0]: rank for rank, row in enumerate(text_rows,  start=1)}

    # Merge all unique doc ids
    all_ids = set(vector_rank) | set(text_rank)

    # Build a lookup from id → row data (prefer vector_rows since they have similarity)
    row_lookup: dict[int, dict] = {}
    for row in vector_rows:
        row_lookup[row[0]] = {
            "source_filename": row[1],
            "section_title": row[2],
            "content": row[3],
            "similarity": row[4],
        }
    for row in text_rows:
        if row[0] not in row_lookup:
            row_lookup[row[0]] = {
                "source_filename": row[1],
                "section_title": row[2],
                "content": row[3],
                "similarity": 0.0,  # No vector score for keyword-only hits
            }

    rrf_scores: list[tuple[float, int]] = []
    for doc_id in all_ids:
        score = 0.0
        if doc_id in vector_rank:
            score += 1.0 / (_RRF_K + vector_rank[doc_id])
        if doc_id in text_rank:
            score += 1.0 / (_RRF_K + text_rank[doc_id])
        rrf_scores.append((score, doc_id))

    # Sort by RRF score descending, take top-k
    rrf_scores.sort(reverse=True)
    top_ids = [doc_id for _, doc_id in rrf_scores[:settings.retrieval_k]]

    # ── Similarity threshold filter ───────────────────────────────────────────
    # Only apply the threshold to vector similarity score (not RRF score) so
    # keyword-only hits are still surfaced when they come from keyword matches.
    results = []
    for doc_id in top_ids:
        chunk = row_lookup[doc_id]
        # Keyword-only hits (similarity=0.0) are kept only if they ranked in text search
        is_keyword_hit = doc_id not in vector_rank and doc_id in text_rank
        if chunk["similarity"] >= settings.similarity_threshold or is_keyword_hit:
            results.append(chunk)

    logger.info(
        "Retrieval: session=%s vector_hits=%d text_hits=%d merged=%d returned=%d top_sim=%.3f",
        session_id, len(vector_rows), len(text_rows), len(all_ids), len(results), top_score,
    )
    return results


async def retrieve_raw(session_id: str, query: str) -> list[dict]:
    """Same as retrieve() but skips the similarity threshold — for the /search debug endpoint.
    Returns raw scores so developers can see exactly what the retriever found.
    """
    results = await retrieve(session_id, query)
    return results