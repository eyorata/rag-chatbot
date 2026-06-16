from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://192.168.6.233:11434"
    llm_provider: str = "ollama"
    llm_model: str = "qwen3.5:latest"
    ollama_base_url: str = "http://192.168.6.233:11434"
    lmstudio_base_url: str = "http://localhost:1234"
    llm_api_key: str = ""
    retrieval_k: int = 5
    similarity_threshold: float = 0.3

    class Config:
        env_file = ".env"

settings = Settings()