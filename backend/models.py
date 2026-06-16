from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    session_id: str
    query: str

class SourceChunk(BaseModel):
    source_filename: str
    section_title: Optional[str]
    similarity: float
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]

class UploadResponse(BaseModel):
    session_id: str
    files: list[str]
    chunks_indexed: int