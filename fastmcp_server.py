### Directory layout (summary):
#
# ├── main.py
# ├── config/settings.py
# ├── routes/rag.py
# ├── services/rag_agent.py
# ├── utils/logger.py
# ├── utils/cursor.py

# -----------------------------
# FILE: main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import asyncio

from config.settings import settings
from utils.logger import log
from utils.cursor import attach_cursor_id
from routes.rag import router as rag_router

logger = log.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting token and tracing services...")
    # Placeholder for background services (token refresh, telemetry, etc.)
    yield
    logger.info("Shutting down background services...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Agent with FastMCP",
        description="Refactored RAG agent using FastAPI MCP pattern",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.middleware("http")(attach_cursor_id)
    app.include_router(rag_router, prefix="/rag", tags=["rag"])
    return app


app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )

# -----------------------------
# FILE: config/settings.py

import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8080))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

settings = Settings()

# -----------------------------
# FILE: routes/rag.py

from fastapi import APIRouter, Request, Depends
from services.rag_agent import rag_chat_handler

router = APIRouter()

@router.post("/query")
async def rag_query(request: Request):
    return await rag_chat_handler(request)

# -----------------------------
# FILE: services/rag_agent.py

from fastapi import Request
from utils.logger import log

logger = log.get_logger()

async def rag_chat_handler(request: Request):
    try:
        body = await request.json()
        query = body.get("query")

        logger.info(f"Received query: {query[:100]}")

        # Stubbed logic (integrate with your RAG agent here)
        response = {
            "answer": f"Stub response for: {query}",
            "source": "knowledge_base",
            "trace_id": request.state.cursor_id
        }
        return response

    except Exception as e:
        logger.error(f"Error processing RAG request: {str(e)}")
        return {"error": str(e)}

# -----------------------------
# FILE: utils/logger.py

import logging
import sys

class Logger:
    @staticmethod
    def get_logger(name="app"):
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

log = Logger()

# -----------------------------
# FILE: utils/cursor.py

import uuid
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware

async def attach_cursor_id(request: Request, call_next):
    cursor_id = str(uuid.uuid4())
    request.state.cursor_id = cursor_id
    response = await call_next(request)
    response.headers["X-Cursor-ID"] = cursor_id
    return response

