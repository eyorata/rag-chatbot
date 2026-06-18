import os
from pydantic_settings import BaseSettings

def _get_env_file() -> str:
    """Determine which env file to load from CONFIG_FILE environment variable."""
    return os.getenv("CONFIG_FILE", ".env")

class Settings(BaseSettings):
    database_url: str = ""
    embedding_model: str = ""
    embedding_base_url: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    ollama_base_url: str = ""
    lmstudio_base_url: str = ""
    llm_api_key: str = ""
    
    # Retrieval
    retrieval_k: int = 5
    similarity_threshold: float = 0.25
    
    # Performance & Limits
    max_file_size_mb: int = 50
    embed_concurrency: int = 5
    
    # Conversational memory
    history_turns: int = 6

    class Config:
        env_file = _get_env_file()

settings = Settings()