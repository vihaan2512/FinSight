from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""

    # Qdrant
    qdrant_db_path: str = "./qdrant_db"
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Relational Database (PostgreSQL / SQLite fallback)
    postgres_url: str = ""
    database_url: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    ingest_interval_minutes: int = 30

    # Groq model
    groq_model: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()