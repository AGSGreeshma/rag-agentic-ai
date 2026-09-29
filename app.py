"""
FastAPI interface for the Agentic AI RAG chatbot.

Run:   uvicorn app:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import build_rag_graph, run_query


# --- Request / response schemas ---
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
    sources: list[Source]


# --- App setup: build the graph once at startup, not on every request ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_rag_graph()
    yield


app = FastAPI(
    title="Agentic AI RAG API",
    description="Answers questions strictly from the Agentic AI eBook using LangGraph + Pinecone.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return run_query(app.state.graph, request.query.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc