from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from core.config import settings

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


@app.on_event("startup")
async def startup_event():
    print(f"[INIT] {settings.PROJECT_NAME} System Shell Online Engine Boot sequence running...")


app.include_router(api_router)
