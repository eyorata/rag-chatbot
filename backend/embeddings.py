import httpx
from config import settings

async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{settings.embedding_base_url}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]