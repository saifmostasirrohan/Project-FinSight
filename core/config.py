from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GROQ_API_KEY: str = Field(default="mock_key")
    SUPABASE_URL: str = Field(default="")
    SUPABASE_KEY: str = Field(default="")
    HF_API_KEY: str = Field(default="")
    EMBEDDING_PROVIDER: str = Field(default="huggingface")
    LLM_PROVIDER: str = Field(default="groq")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/v1")
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="DEBUG")

    PROJECT_NAME: str = "FinSight API"
    VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
