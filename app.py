"""
FastAPI backend for the Agentic AI Research Assistant.

Run:   uvicorn app:app --reload
Docs:  http://127.0.0.1:8000/docs

Endpoints
    POST /chat          -> answer + retrieved chunks + confidence (assignment format)
    GET  /chat/stream   -> same pipeline, streamed as Server-Sent Events (one event per LangGraph node)
    GET  /health        -> live checks: Pinecone reachable, LLM credentials valid
    GET  /kb/info       -> knowledge-base metadata (pages, chunks, models)
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src import config
from src.graph import build_rag_graph, format_result, initial_state, run_query

NODE_NAMES = {"retrieve", "grade_documents", "rewrite_query", "generate",
              "check_grounding", "finalize", "refuse"}
HEALTH_TTL_SECONDS = 30

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
def _pdf_page_counts() -> tuple[Optional[int], Optional[int]]:
    try:
        reader = PdfReader(str(config.PDF_PATH))
        with_text = sum(1 for p in reader.pages
                        if len((p.extract_text() or "").strip()) >= config.MIN_PAGE_CHARS)
        return len(reader.pages), with_text
    except Exception:
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_rag_graph()
    app.state.index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX_NAME)
    app.state.pdf_pages = _pdf_page_counts()
    app.state.health_cache = (0.0, None)
    yield


app = FastAPI(
    title="Agentic AI Research Assistant API",
    description="Answers questions strictly from the Agentic AI eBook using LangGraph + Pinecone.",
    version="1.1.0",
    lifespan=lifespan,
)


# --- Health / metadata ---
def _check_health() -> dict:
    """Real checks, cached briefly so the UI can poll without hammering Pinecone/OpenAI."""
    now = time.monotonic()
    cached_at, cached = app.state.health_cache
    if cached and now - cached_at < HEALTH_TTL_SECONDS:
        return cached

    result = {"pinecone": False, "llm": False, "vector_count": None, "errors": {}}
    try:
        stats = app.state.index.describe_index_stats()
        result["pinecone"] = True
        result["vector_count"] = stats.total_vector_count
    except Exception as exc:
        result["errors"]["pinecone"] = str(exc)[:300]
    try:
        # Free metadata call: confirms the key is valid and the model is available
        OpenAI(api_key=config.OPENAI_API_KEY).models.retrieve(config.LLM_MODEL)
        result["llm"] = True
    except Exception as exc:
        result["errors"]["llm"] = str(exc)[:300]

    app.state.health_cache = (now, result)
    return result


@app.get("/health")
def health():
    h = _check_health()
    return {
        "status": "ok" if h["pinecone"] and h["llm"] else "degraded",
        "api": True,
        "pinecone": h["pinecone"],
        "llm": h["llm"],
        "errors": h["errors"],
    }


@app.get("/kb/info")
def kb_info():
    h = _check_health()
    pages, pages_with_text = app.state.pdf_pages
    return {
        "document": config.PDF_PATH.name,
        "pages": pages,
        "pages_with_text": pages_with_text,
        "chunks": h["vector_count"],
        "vector_db": "Pinecone",
        "index_name": config.PINECONE_INDEX_NAME,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "top_k": config.TOP_K,
    }


# --- Chat ---
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return run_query(app.state.graph, request.query.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc


def _describe(node: str, state: dict) -> str:
    """Short progress detail for each node, shown in the UI's pipeline view."""
    if node == "retrieve":
        return f"{len(state['chunks'])} chunks"
    if node == "grade_documents":
        return f"{sum(c['relevant'] for c in state['chunks'])} relevant"
    if node == "rewrite_query":
        return f"{len(state['search_queries'])} new queries"
    if node == "check_grounding":
        return f"grounding {state['grounding_score']:.2f}"
    if node == "finalize":
        return f"confidence {state['confidence_score']:.2f}"
    return ""


def _task_started(chunk) -> Optional[str]:
    """Extract the node name from a LangGraph 'debug' task-start event.

    The debug payload shape is not part of LangGraph's stable API, so a version bump can change it.
    Warn rather than swallow: otherwise the UI's pipeline view just silently stops populating."""
    if not isinstance(chunk, dict) or chunk.get("type") != "task":
        return None
    payload = chunk.get("payload")
    if not isinstance(payload, dict):
        log.warning("LangGraph debug task event had no payload dict (keys=%s); "
                    "pipeline view will not show node starts.", sorted(chunk))
        return None
    name = payload.get("name")
    return name if name in NODE_NAMES else None


@app.get("/chat/stream")
def chat_stream(query: str = Query(..., min_length=1, max_length=1000)):
    """Server-Sent Events: {"type":"start"|"done", "node":...} per node, then {"type":"result", "data":...}."""
    question = query.strip()

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def events():
        state = initial_state(question)
        started = time.perf_counter()
        try:
            for mode, chunk in app.state.graph.stream(state, stream_mode=["updates", "debug"]):
                if mode == "debug":
                    node = _task_started(chunk)
                    if node:
                        yield sse({"type": "start", "node": node})
                elif mode == "updates":
                    for node, changes in chunk.items():
                        if node not in NODE_NAMES or not isinstance(changes, dict):
                            continue
                        state.update(changes)
                        yield sse({"type": "done", "node": node, "detail": _describe(node, state)})
            result = format_result(question, state)
            result["latency"] = round(time.perf_counter() - started, 2)
            yield sse({"type": "result", "data": result})
        except Exception as exc:
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
