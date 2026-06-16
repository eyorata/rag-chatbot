import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — tuned for nomic-embed-text (768-dim, ~350 token sweet spot)
# ─────────────────────────────────────────────────────────────────────────────
MIN_WORDS = 30      # Chunks shorter than this carry too little signal
MAX_WORDS = 500     # Hard ceiling before force-splitting
TARGET_WORDS = 350  # Ideal chunk size
OVERLAP_SENTENCES = 2  # Sentences carried forward as overlap context


# ─────────────────────────────────────────────────────────────────────────────
# Section-header detection for non-markdown documents (PDF, DOCX, TXT)
# ─────────────────────────────────────────────────────────────────────────────

_ALLCAPS_RE = re.compile(r'^[A-Z0-9 \-/:&,\.]{3,80}$')
_SHORT_LINE_RE = re.compile(r'^.{1,79}$')

def _is_section_header(line: str) -> bool:
    """Heuristic: short all-caps line or short standalone line looks like a heading.
    This catches 'INTRODUCTION', 'Chapter 3: Results', etc. from PDF text extraction.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    # All-caps lines are strong signals (e.g. 'ABSTRACT', 'METHODOLOGY')
    if _ALLCAPS_RE.match(stripped):
        return True
    # Short lines that end without sentence punctuation are likely headings
    if len(stripped) < 60 and not stripped.endswith(('.', ',', ';', ':', '?', '!')):
        return True
    return False


def _extract_sections(text: str) -> list[tuple[str, str]]:
    """Split raw text into (title, body) pairs using header heuristics.
    Returns at least one section so downstream logic always has a title.
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_body: list[str] = []

    for line in lines:
        if _is_section_header(line):
            # Flush accumulated body as a section before starting a new one
            body = "\n".join(current_body).strip()
            if body:
                sections.append((current_title, body))
            current_title = line.strip()
            current_body = []
        else:
            current_body.append(line)

    # Flush final section
    body = "\n".join(current_body).strip()
    if body:
        sections.append((current_title, body))

    # Fallback: no headers detected — treat whole document as one section
    if not sections:
        sections = [("Document", text.strip())]

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-boundary aware splitting
# ─────────────────────────────────────────────────────────────────────────────

# Matches sentence-ending punctuation followed by whitespace+capital, handling
# common abbreviations by requiring at least one space and a capital letter after.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\(\[])')

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries.
    Falls back gracefully for texts without sentence-ending punctuation.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _sentences_to_chunks(sentences: list[str], title: str) -> list[dict]:
    """Accumulate sentences into word-bounded chunks with sentence-level overlap.

    Why sentence overlap instead of word overlap: carrying full sentences avoids
    starting a chunk mid-thought, which confuses the embedding model.
    """
    chunks: list[dict] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())

        # Force-flush an overlong sentence by itself (rare, but PDF tables can do this)
        if sent_words > MAX_WORDS:
            if current:
                _flush_chunk(current, title, chunks)
                current = []
                current_words = 0
            # Hard-split the giant sentence at word boundaries
            words = sent.split()
            for i in range(0, len(words), MAX_WORDS):
                piece = " ".join(words[i:i + MAX_WORDS])
                if len(piece.split()) >= MIN_WORDS:
                    chunks.append({"title": title, "content": piece})
            continue

        if current_words + sent_words > MAX_WORDS and current:
            _flush_chunk(current, title, chunks)
            # Carry last N sentences forward as overlap context
            current = current[-OVERLAP_SENTENCES:]
            current_words = sum(len(s.split()) for s in current)

        current.append(sent)
        current_words += sent_words

    if current:
        _flush_chunk(current, title, chunks)

    return chunks


def _flush_chunk(sentences: list[str], title: str, out: list[dict]) -> None:
    """Emit a chunk only if it meets the minimum word count threshold.
    Sub-30-word chunks are usually headers, captions, or noise — not useful for RAG.
    """
    content = " ".join(sentences).strip()
    if len(content.split()) >= MIN_WORDS:
        out.append({"title": title, "content": content})
    else:
        logger.debug("Dropped short chunk (%d words) from section '%s'",
                     len(content.split()), title)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown-specific chunking (unchanged strategy, improved internals)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_markdown(text: str) -> list[dict]:
    """Split markdown on ## / ### headers, then sentence-chunk each section."""
    raw_sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
    chunks: list[dict] = []
    for section in raw_sections:
        title_match = re.match(r'^#{1,3} (.+)', section)
        title = title_match.group(1).strip() if title_match else "Section"
        sentences = _split_sentences(section)
        chunks.extend(_sentences_to_chunks(sentences, title))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, filename: str) -> list[dict]:
    """Entry point: route to the right chunking strategy based on file extension.

    Returns a list of {"title": str, "content": str} dicts ready for embedding.
    """
    if filename.lower().endswith(".md"):
        return chunk_markdown(text)

    # For PDF/DOCX/TXT: detect section headers, then sentence-chunk each section
    sections = _extract_sections(text)
    chunks: list[dict] = []
    for title, body in sections:
        sentences = _split_sentences(body)
        chunks.extend(_sentences_to_chunks(sentences, title))

    if not chunks:
        logger.warning("No chunks produced from file '%s' — document may be empty", filename)

    return chunks