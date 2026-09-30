"""API service layer: the only place the UI talks to the backend.

    RAG_API_URL   base URL of the FastAPI backend (default http://127.0.0.1:8000)
    RAG_API_MODE  "live" (default) or "mock" to develop the UI without a backend

Both clients expose the same interface, so switching is a one-line change:
    health()      -> SystemStatus
    kb_info()     -> KBInfo | None
    stream(query) -> iterator of pipeline events, ending with {"type": "result", "response": RAGResponse}
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator, Optional, Protocol

import httpx

from ui.models import KBInfo, RAGResponse, SystemStatus

DEFAULT_API_URL = "http://127.0.0.1:8000"


class APIError(Exception):
    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint


class RAGClient(Protocol):
    def health(self) -> SystemStatus: ...
    def kb_info(self) -> Optional[KBInfo]: ...
    def stream(self, query: str) -> Iterator[dict]: ...


# ---------------------------------------------------------------------------
# Live client (FastAPI backend)
# ---------------------------------------------------------------------------
class HttpRAGClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout, connect=5.0)

    def _offline_error(self) -> APIError:
        return APIError(
            f"Can't reach the RAG API at {self.base_url}.",
            hint="Start the backend in another terminal:  uvicorn app:app --reload",
        )

    def health(self) -> SystemStatus:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=httpx.Timeout(15.0, connect=3.0))
            r.raise_for_status()
            data = r.json()
            return SystemStatus(mode="live", api=True,
                                pinecone=data.get("pinecone"), llm=data.get("llm"))
        except httpx.HTTPError as exc:
            return SystemStatus(mode="live", api=False, error=str(exc))

    def kb_info(self) -> Optional[KBInfo]:
        try:
            r = httpx.get(f"{self.base_url}/kb/info", timeout=httpx.Timeout(15.0, connect=3.0))
            r.raise_for_status()
            return KBInfo.from_api(r.json())
        except httpx.HTTPError:
            return None

    def ask(self, query: str) -> RAGResponse:
        """Plain POST /chat (no progress events)."""
        try:
            r = httpx.post(f"{self.base_url}/chat", json={"query": query}, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise self._offline_error() from exc
        except httpx.TimeoutException as exc:
            raise APIError("The request timed out.", hint="The LLM provider may be slow. Try again.") from exc
        if r.status_code >= 400:
            raise APIError(f"The API returned an error ({r.status_code}).", hint=_detail(r))
        return RAGResponse.from_api(r.json())

    def stream(self, query: str) -> Iterator[dict]:
        """GET /chat/stream (Server-Sent Events): one event per LangGraph node, then the result.
        Falls back to POST /chat if the backend has no streaming endpoint."""
        url = f"{self.base_url}/chat/stream"
        try:
            with httpx.stream("GET", url, params={"query": query}, timeout=self.timeout) as r:
                if r.status_code == 404:
                    yield {"type": "result", "response": self.ask(query)}
                    return
                if r.status_code >= 400:
                    r.read()
                    raise APIError(f"The API returned an error ({r.status_code}).", hint=_detail(r))
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    if event.get("type") == "error":
                        raise APIError("The RAG pipeline failed.", hint=event.get("message"))
                    if event.get("type") == "result":
                        yield {"type": "result", "response": RAGResponse.from_api(event["data"])}
                        return
                    yield event
        except httpx.ConnectError as exc:
            raise self._offline_error() from exc
        except httpx.TimeoutException as exc:
            raise APIError("The request timed out.", hint="The LLM provider may be slow. Try again.") from exc
        raise APIError("The stream ended before a result arrived.", hint="Check the backend logs.")


def _detail(r: httpx.Response) -> Optional[str]:
    try:
        return str(r.json().get("detail"))
    except Exception:
        return r.text[:300] or None


# ---------------------------------------------------------------------------
# Mock client (UI development without a backend) - clearly labelled sample data
# ---------------------------------------------------------------------------
class MockRAGClient:
    """Returns clearly labelled sample data. Never used unless RAG_API_MODE=mock."""

    _IN_SCOPE_WORDS = ("agent", "agentic", "ai", "llm", "memory", "architect", "use case", "challenge")

    def health(self) -> SystemStatus:
        return SystemStatus(mode="mock", api=False, pinecone=None, llm=None)

    def kb_info(self) -> Optional[KBInfo]:
        return None  # never fabricate knowledge-base metadata

    def stream(self, query: str) -> Iterator[dict]:
        in_scope = any(w in query.lower() for w in self._IN_SCOPE_WORDS)
        steps = [("retrieve", "5 chunks"), ("grade_documents", "3 relevant" if in_scope else "0 relevant")]
        steps += [("generate", ""), ("check_grounding", "grounding 1.00"), ("finalize", "confidence 0.86")] \
            if in_scope else [("refuse", "")]
        for node, detail in steps:
            yield {"type": "start", "node": node}
            time.sleep(0.5)
            yield {"type": "done", "node": node, "detail": detail}

        chunks = [
            "[Mock passage] Agentic AI refers to systems that pursue goals autonomously, planning and "
            "acting with limited human supervision.",
            "[Mock passage] Core components include perception, memory, planning, decision-making and "
            "action layers.",
            "[Mock passage] Unrelated sample text that was retrieved but not used in the answer.",
        ]
        data = {
            "query": query,
            "final_answer": (
                "**[Mock response]** Agentic AI describes systems that plan and act autonomously toward a goal "
                "(p. 7). Typical components are:\n- Perception and memory (p. 19)\n- Planning and decision-making (p. 32)"
                if in_scope else "I cannot answer this based on the provided document."
            ),
            "retrieved_context_chunks": chunks,
            "confidence_score": 0.86 if in_scope else 0.0,
            "sources": [
                {"page": 7, "similarity": 0.73, "used_in_answer": in_scope},
                {"page": 19, "similarity": 0.69, "used_in_answer": in_scope},
                {"page": 42, "similarity": 0.07 if not in_scope else 0.41, "used_in_answer": False},
            ],
            "out_of_scope": not in_scope,
            "grounding_score": 1.0 if in_scope else None,
            "retrieval_score": 0.71 if in_scope else None,
            "latency": 2.5,
        }
        yield {"type": "result", "response": RAGResponse.from_api(data)}


def get_client() -> RAGClient:
    if os.getenv("RAG_API_MODE", "live").lower() == "mock":
        return MockRAGClient()
    return HttpRAGClient(os.getenv("RAG_API_URL", DEFAULT_API_URL))
