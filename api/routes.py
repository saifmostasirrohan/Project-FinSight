import structlog
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from langchain_core.messages import HumanMessage

from api.schemas import HealthResponse, ChatRequest, ChatResponse, UploadResponse
from agents.graph import compiled_graph
from core.config import settings

router = APIRouter()
logger = structlog.get_logger()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def get_health():
    return HealthResponse(status="ok", version=settings.VERSION, env=settings.APP_ENV)


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def post_chat(payload: ChatRequest):
    structlog.contextvars.bind_contextvars(session_id=payload.session_id)
    logger.info("agent_pipeline_invocation_started", user_query_len=len(payload.message))

    try:
        input_message = HumanMessage(content=payload.message)
        initial_state: dict = {
            "messages": [input_message],
            "session_id": payload.session_id,
            "current_agent": "initialization",
            "documents_loaded": False,
            "audit_findings": [],
            "user_confirmed": False,
        }

        output_state = await compiled_graph.ainvoke(initial_state)
        final_message = output_state["messages"][-1]

        logger.info(
            "agent_pipeline_invocation_successful",
            resolved_agent=output_state.get("current_agent"),
        )

        return ChatResponse(
            response=str(final_message.content),
            session_id=payload.session_id,
            agent_used=output_state.get("current_agent", "unknown_agent"),
        )
    except Exception as exc:
        logger.error("agent_pipeline_invocation_failed", error_detail=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Runtime Execution Error: {str(exc)}",
        )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def post_upload(file: UploadFile = File(...)):
    logger.info(
        "document_upload_request_received",
        filename=file.filename,
        content_type=file.content_type,
    )

    allowed_types = ["application/pdf", "text/csv"]
    if file.content_type not in allowed_types:
        logger.warning(
            "document_upload_rejected_invalid_type",
            content_type=file.content_type,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {file.content_type}. Only PDF and CSV are permitted.",
        )

    content = await file.read()
    size = len(content)

    logger.info("document_upload_accepted", filename=file.filename, size_bytes=size)
    return UploadResponse(
        filename=file.filename or "unknown_file",
        size_bytes=size,
        file_type=file.content_type,
        status="accepted",
    )
