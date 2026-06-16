SYSTEM_PROMPT = """You are a helpful expert assistant. Your job is to answer the user's question using ONLY the provided context.

RULES:
1. If the context does not contain enough information to answer, say so clearly instead of guessing or fabricating information.
2. When you use information from the context, you MUST cite which source file and section it came from (e.g., "According to document.pdf in the Introduction section...").
3. Give clear, direct answers formatted nicely with Markdown where appropriate.
4. Do not mention the word "context" in your response to the user.

AVAILABLE DOCUMENTS:
{available_docs}"""

def build_messages(query: str, chunks: list[dict], history: list[dict] = None, available_docs: list[str] = None) -> list[dict]:
    # Build document inventory
    docs_text = "\n".join(f"- {doc}" for doc in available_docs) if available_docs else "(None available)"
    sys_prompt = SYSTEM_PROMPT.format(available_docs=docs_text)
    
    # Build context string from retrieved chunks
    if not chunks:
        context = "(no relevant context was found for this query)"
    else:
        context = "\n\n".join(
            f"--- SOURCE: {c['source_filename']} (Section: {c.get('section_title') or 'Unknown'}) ---\n{c['content']}"
            for c in chunks
        )
    
    # Construct final messages list
    messages = [{"role": "system", "content": sys_prompt}]
    
    # Inject conversational history if available
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    # Add current query with context
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    messages.append({"role": "user", "content": user_content})
    
    return messages