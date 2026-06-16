from pydantic import BaseModel, Field
from typing import Optional, List

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
    sources: List[SourceChunk]
    chunks_retrieved: int
    top_similarity: float

class UploadResponse(BaseModel):
    session_id: str
    files_processed: List[str]
    files_failed: List[str]
    chunks_indexed: int
    chunks_failed: int
    errors: List[str]

class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    upload_timestamp: str
    file_size_bytes: Optional[int]

class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class SearchResponse(BaseModel):
    chunks: List[SourceChunk]
    
class ModelList(BaseModel):
    llm_models: List[str]
    embed_models: List[str]