import httpx
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

async def embed(text: str) -> Optional[list[float]]:
    """Fetches an embedding for a text chunk. returns None on failure instead of crashing."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{settings.embedding_base_url}/api/embeddings",
                json={"model": settings.embedding_model, "prompt": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None