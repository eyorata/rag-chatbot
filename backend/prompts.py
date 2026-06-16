SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly instead of guessing.
When you use information from the context, mention which source file it came from."""

def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        context = "(no relevant context was found)"
    else:
        context = "\n\n".join(
            f"Source: {c['source_filename']} ({c.get('section_title') or 'section'})\n{c['content']}"
            for c in chunks
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]