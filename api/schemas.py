from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique context tracker ID string",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User string query block",
    )

    @field_validator("session_id", "message")
    @classmethod
    def prevent_empty_spaces(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot contain only whitespace characters")
        return value.strip()


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent_used: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    file_type: str
    status: str
