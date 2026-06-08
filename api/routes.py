import structlog
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from langchain_core.messages import HumanMessage

from api.schemas import HealthResponse, ChatRequest, ChatResponse
from agents.graph import compiled_graph
from core.config import settings
from services.database import SupabaseManager
from services.parser import DocumentParser

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


@router.post("/upload", response_model=dict, status_code=status.HTTP_201_CREATED)
async def post_upload(file: UploadFile = File(...), session_id: str = "default_session"):
    structlog.contextvars.bind_contextvars(session_id=session_id)
    logger.info(
        "document_upload_request_received",
        filename=file.filename,
        content_type=file.content_type,
    )

    allowed_types = {"application/pdf": "PDF", "text/csv": "CSV"}
    if file.content_type not in allowed_types:
        logger.warning(
            "document_upload_rejected_invalid_type",
            content_type=file.content_type,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file format protocol: {file.content_type}. Only PDF and CSV are permitted.",
        )

    try:
        file_bytes = await file.read()

        if file.content_type == "application/pdf":
            parsed_blocks = DocumentParser.parse_pdf(file_bytes)
        else:
            parsed_blocks = DocumentParser.parse_csv(file_bytes)

        supabase = SupabaseManager.get_client()
        db_record = {
            "session_id": session_id,
            "filename": file.filename or "unknown_file",
            "file_type": allowed_types[file.content_type],
            "total_chunks": len(parsed_blocks),
        }

        db_res = supabase.table("documents").insert(db_record).execute()
        if not db_res.data:
            raise RuntimeError(
                "Cloud ledger registration failure occurred during insertion phase."
            )

        generated_doc_id = db_res.data[0]["id"]
        logger.info(
            "document_upload_extraction_complete",
            document_id=generated_doc_id,
            parsed_elements_count=len(parsed_blocks),
        )

        return {
            "document_id": generated_doc_id,
            "filename": file.filename,
            "parsed_elements_count": len(parsed_blocks),
            "status": "extraction_complete_ledger_registered",
            "preview_sample": parsed_blocks[:2] if parsed_blocks else [],
        }
    except Exception as exc:
        logger.error("document_processing_pipeline_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document parsing error occurred: {str(exc)}",
        )
