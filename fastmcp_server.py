from __future__ import annotations

"""mcp_server.py
=================
Minimal Model Context Protocol (MCP) server with API‑level tracing and
Cursor‑ID support, ready to plug into a LangGraph‑powered RAG pipeline.

❯  Development:
    uvicorn mcp_server:app --reload

❯  Production (Docker/VM):
    uvicorn mcp_server:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
- OPENAI_API_KEY (if your RAG agent needs it)
- MCP_LOG_DIR       optional, default "logs/"

Replace the stubbed `rag_agent()` with your own LangGraph chain or any async
function that returns `(answer: str, docs: list[str])`.
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.getenv("MCP_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mcp_server.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("mcp")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def generate_cursor(step: str) -> str:
    """Return a globally‑unique cursor ID such as `query_20250622T184512123Z_a1b2c3d4`."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{step}_{ts}_{uuid.uuid4().hex[:8]}"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural‑language question or prompt")
    top_k: int = Field(3, ge=1, description="Number of passages to retrieve")

class QueryResponse(BaseModel):
    cursor_id: str
    answer: str
    docs: List[str]
    latency_ms: int

# ---------------------------------------------------------------------------
# RAG agent stub – plug in your LangGraph chain here
# ---------------------------------------------------------------------------

async def rag_agent(query: str, top_k: int) -> tuple[str, List[str]]:
    """Stubbed async RAG agent.

    Replace with: `return await my_chain.arun({"query": query, "top_k": top_k})`
    """
    await asyncio.sleep(0.05)  # simulate network/compute latency
    return "Stub answer – replace with real RAG output", [f"doc {i}" for i in range(1, top_k + 1)]

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MCP Server with API Tracking",
    version="0.1.0",
    description="Serves RAG answers and exposes Cursor‑ID‑based tracing.",
)

@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    cursor_id = generate_cursor("request")
    request.state.cursor_id = cursor_id
    start = time.perf_counter()

    logger.info(f"{cursor_id} | IN  | {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"{cursor_id} | ERR | {exc}")
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    response.headers["X-Cursor-ID"] = cursor_id
    logger.info(
        f"{cursor_id} | OUT | status={response.status_code} | {duration_ms:.2f}ms",
    )
    return response


@app.get("/health", tags=["Utility"])
async def health():
    """Liveness endpoint for load‑balancers and monitoring."""
    return {"status": "ok", "utc": datetime.utcnow().isoformat()}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(payload: QueryRequest):
    cursor_id = generate_cursor("query")
    logger.info(f"{cursor_id} | query='{payload.query}' | top_k={payload.top_k}")

    start = time.perf_counter()
    try:
        answer, docs = await rag_agent(payload.query, payload.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"{cursor_id} | rag_agent failed: {exc}")
        raise HTTPException(status_code=500, detail="RAG agent failure") from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(f"{cursor_id} | success | latency_ms={latency_ms}")
    return QueryResponse(cursor_id=cursor_id, answer=answer, docs=docs, latency_ms=latency_ms)


@app.get("/trace/{cursor_id}", tags=["Tracing"])
async def trace_endpoint(cursor_id: str):
    """Return all log lines that contain the requested Cursor ID."""
    log_path = os.path.join(LOG_DIR, "mcp_server.log")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    with open(log_path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if cursor_id in line]

    if not lines:
        raise HTTPException(status_code=404, detail="Cursor ID not found in logs")

    return {"cursor_id": cursor_id, "events": lines}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)
