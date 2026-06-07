import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from core.config import settings
from core.logging import configure_logging

configure_logging()
logger = structlog.get_logger()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Autonomous Financial Audit & Fraud Detection System",
    version=settings.VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def append_telemetry_tracing_context(request: Request, call_next):
    """
    Captures individual incoming HTTP actions, injects distinct tracking tokens,
    and isolates transaction performance profiles across service nodes.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    logger.info("http_request_received")
    response = await call_next(request)

    duration = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        "http_request_completed",
        status_code=response.status_code,
        duration_ms=duration,
    )

    response.headers["X-Request-ID"] = request_id
    return response


@app.on_event("startup")
async def startup_event():
    logger.info(
        "system_boot_sequence_complete",
        project_name=settings.PROJECT_NAME,
        env=settings.APP_ENV,
        version=settings.VERSION,
    )


app.include_router(api_router)
