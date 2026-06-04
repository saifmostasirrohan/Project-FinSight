from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas import ChatRequest, ChatResponse, HealthResponse, UploadResponse
from core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def get_health():
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        env=settings.APP_ENV,
    )


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def post_chat(payload: ChatRequest):
    return ChatResponse(
        response=f"Infrastructure Shell Verification Ack. Received: '{payload.message}'",
        session_id=payload.session_id,
        agent_used="shell_placeholder",
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def post_upload(file: UploadFile = File(...)):
    allowed_types = ["application/pdf", "text/csv"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {file.content_type}. Only PDF and CSV are permitted.",
        )

    content = await file.read()
    size = len(content)
    await file.seek(0)

    return UploadResponse(
        filename=file.filename or "unknown_file",
        size_bytes=size,
        file_type=file.content_type,
        status="accepted",
    )
