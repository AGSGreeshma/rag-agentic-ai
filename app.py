"""
FastAPI backend for the Agentic AI Research Assistant.

Run:   uvicorn app:app --reload
Docs:  http://127.0.0.1:8000/docs

Endpoints
    POST /chat          -> answer + retrieved chunks + confidence (assignment format)
    GET  /chat/stream   -> same pipeline, streamed as Server-Sent Events (one event per LangGraph node)
    GET  /health        -> live checks: Pinecone reachable, LLM credentials valid
    GET  /kb/info       -> knowledge-base metadata (pages, chunks, models)

The pipeline itself lives in src/engine.py, which the Streamlit UI can also call directly when
no HTTP backend is running (see ui/api_client.InProcessRAGClient). This module is the HTTP skin
over that engine: request validation, response schemas and SSE framing.
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.engine import EngineError, get_engine

log = logging.getLogger("uvicorn.error")


# --- Schemas ---
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, examples=["What is Agentic AI?"])


class Source(BaseModel):
    page: int
    similarity: float
    used_in_answer: bool


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float
    sources: list[Source] = []
    rewritten_queries: list[str] = []
    out_of_scope: bool = False
    grounding_score: Optional[float] = None
    retrieval_score: Optional[float] = None


# --- Startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the graph, the Pinecone handle and the PDF stats now rather than on the first
    # request, so a misconfigured backend fails at startup instead of mid-question.
    app.state.engine = get_engine()
    app.state.engine.warm()
    yield


app = FastAPI(
    title="Agentic AI Research Assistant API",
    description="Answers questions strictly from the Agentic AI eBook using LangGraph + Pinecone.",
    version="1.1.0",
    lifespan=lifespan,
)


# --- Health / metadata ---
@app.get("/health")
def health():
    return app.state.engine.status()


@app.get("/kb/info")
def kb_info():
    return app.state.engine.kb_info()


# --- Chat ---
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return app.state.engine.ask(request.query.strip())
    except EngineError as exc:
        raise HTTPException(status_code=500, detail=exc.message) from exc


@app.get("/chat/stream")
def chat_stream(query: str = Query(..., min_length=1, max_length=1000)):
    """Server-Sent Events: {"type":"start"|"done", "node":...} per node, then {"type":"result", "data":...}."""
    question = query.strip()
    engine = app.state.engine

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def events():
        for event in engine.stream(question):
            yield sse(event)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
