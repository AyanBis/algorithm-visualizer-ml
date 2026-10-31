"""Backend configuration."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"

    # Experiment settings
    MAX_CONCURRENT_EXPERIMENTS: int = 10
    EXPERIMENT_TTL_SECONDS: int = 3600  # 1 hour
    EVENT_BUFFER_SIZE: int = 1000

    # ML
    DEFAULT_RANDOM_STATE: int = 42

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()