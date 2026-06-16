from pypdf import PdfReader
from docx import Document
import io

def parse_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def parse_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def parse_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_text,
    ".md": parse_text,
}

def parse_file(filename: str, file_bytes: bytes) -> str:
    ext = "." + filename.lower().rsplit(".", 1)[-1]
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return PARSERS[ext](file_bytes)