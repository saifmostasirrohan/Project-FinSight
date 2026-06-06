from fastapi import APIRouter, UploadFile, File, HTTPException, status

from api.schemas import HealthResponse, ChatRequest, ChatResponse, UploadResponse
from core.config import settings
from services import llm_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def get_health():
    return HealthResponse(status="ok", version=settings.VERSION, env=settings.APP_ENV)


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def post_chat(payload: ChatRequest):
    try:
        llm_out = await llm_service.get_response(message=payload.message)
        return ChatResponse(
            response=llm_out,
            session_id=payload.session_id,
            agent_used=f"direct_llm_{settings.LLM_PROVIDER}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def post_upload(file: UploadFile = File(...)):
    allowed_types = ["application/pdf", "text/csv"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {file.content_type}. Only PDF and CSV are permitted.",
        )
    content = await file.read()
    return UploadResponse(
        filename=file.filename or "unknown_file",
        size_bytes=len(content),
        file_type=file.content_type,
        status="accepted",
    )
