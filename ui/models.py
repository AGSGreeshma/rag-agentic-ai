"""Typed data models shared by the API client and the UI components.

The UI never touches raw JSON: every backend response is normalised into these
dataclasses first, so the components work with the minimal response format
(query / final_answer / retrieved_context_chunks / confidence_score) as well as
the richer one our FastAPI backend returns (page numbers, similarity, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Must match REFUSAL in src/graph.py. Only used as a fallback when the backend
# does not send an explicit "out_of_scope" flag.
REFUSAL_HINT = "cannot answer this based on the provided document"


@dataclass
class Source:
    text: str
    page: Optional[int] = None
    similarity: Optional[float] = None
    used_in_answer: bool = True


@dataclass
class RAGResponse:
    query: str
    answer: str
    confidence: float
    sources: list[Source]
    out_of_scope: bool = False
    grounding_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    rewritten_queries: list[str] = field(default_factory=list)
    latency: Optional[float] = None

    @property
    def used_sources(self) -> list[Source]:
        return [s for s in self.sources if s.used_in_answer]

    @property
    def other_sources(self) -> list[Source]:
        return [s for s in self.sources if not s.used_in_answer]

    @classmethod
    def from_api(cls, data: dict) -> "RAGResponse":
        chunks = data.get("retrieved_context_chunks") or []
        meta = data.get("sources") or []
        sources = []
        for i, text in enumerate(chunks):
            m = meta[i] if i < len(meta) else {}
            sources.append(Source(
                text=text,
                page=m.get("page"),
                similarity=m.get("similarity"),
                used_in_answer=m.get("used_in_answer", True),
            ))
        # Sort: passages used in the answer first, then by similarity
        sources.sort(key=lambda s: (not s.used_in_answer, -(s.similarity or 0)))

        answer = data.get("final_answer", "")
        confidence = float(data.get("confidence_score") or 0.0)
        if "out_of_scope" in data:
            out_of_scope = bool(data["out_of_scope"])
        else:
            out_of_scope = confidence == 0.0 and REFUSAL_HINT in answer.lower()

        return cls(
            query=data.get("query", ""),
            answer=answer,
            confidence=confidence,
            sources=sources,
            out_of_scope=out_of_scope,
            grounding_score=data.get("grounding_score"),
            retrieval_score=data.get("retrieval_score"),
            rewritten_queries=data.get("rewritten_queries") or [],
            latency=data.get("latency"),
        )


@dataclass
class KBInfo:
    document: Optional[str] = None
    pages: Optional[int] = None
    pages_with_text: Optional[int] = None
    chunks: Optional[int] = None
    vector_db: Optional[str] = None
    index_name: Optional[str] = None
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    top_k: Optional[int] = None

    @classmethod
    def from_api(cls, data: dict) -> "KBInfo":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class SystemStatus:
    """What the backend has actually confirmed. None means 'unknown'."""
    mode: str                     # "live" | "mock"
    api: bool = False
    pinecone: Optional[bool] = None
    llm: Optional[bool] = None
    error: Optional[str] = None

    @property
    def online(self) -> bool:
        return self.api and bool(self.pinecone) and bool(self.llm)
