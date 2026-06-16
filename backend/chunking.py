import re

def chunk_markdown(text: str, max_words: int = 350, overlap: int = 50) -> list[dict]:
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    chunks = []
    for section in sections:
        title_match = re.match(r"^#{1,3} (.+)", section)
        title = title_match.group(1).strip() if title_match else "Untitled"
        chunks.extend(
            {"title": title, "content": c}
            for c in _sliding_window(section, max_words, overlap)
        )
    return chunks

def _sliding_window(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), step)
        if words[i:i + max_words]
    ]

def chunk_text(text: str, filename: str) -> list[dict]:
    if filename.lower().endswith(".md"):
        return chunk_markdown(text)
    return [{"title": filename, "content": c} for c in _sliding_window(text, 350, 50)]