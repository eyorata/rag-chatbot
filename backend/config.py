from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"
    llm_provider: str = "ollama"
    llm_model: str = "qwen3.5:latest"
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234"
    llm_api_key: str = ""
    
    # Retrieval
    retrieval_k: int = 5
    similarity_threshold: float = 0.25  # lowered from 0.3 for hybrid search safety
    
    # Performance & Limits
    max_file_size_mb: int = 50
    embed_concurrency: int = 5  # Max concurrent calls to embedding model
    
    # Conversational memory
    history_turns: int = 6  # Number of past messages to include in prompt

    class Config:
        env_file = ".env"

settings = Settings()