"""API service layer: the only place the UI talks to the backend.

    RAG_API_URL   base URL of a running FastAPI backend. Set it to use HTTP mode;
                  leave it unset to run the RAG pipeline inside this process.
    RAG_API_MODE  "live" (default) or "mock" to develop the UI without a backend

Three interchangeable clients, all with the same interface:
    health()      -> SystemStatus
    kb_info()     -> KBInfo | None
    ask(query)    -> RAGResponse
    stream(query) -> iterator of pipeline events, ending with {"type": "result", "response": RAGResponse}

    HttpRAGClient       talks to uvicorn over HTTP/SSE          (local development, any API host)
    InProcessRAGClient  calls src/engine.py directly            (Streamlit Community Cloud)
    MockRAGClient       sample data, never real                 (UI work without credentials)

Importing src.config here (rather than only inside the in-process client) is deliberate: it is
what loads Streamlit Secrets into the environment, and RAG_API_URL / RAG_API_MODE themselves may
come from there, so it has to happen before get_client() reads them.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator, Optional, Protocol

import httpx

from src import config  # noqa: F401  - imported for its Streamlit Secrets side effect
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


def _hint_for(kind: Optional[str]) -> Optional[str]:
    """Turn the engine's error `kind` into something the user can act on."""
    return {
        "config": "An API key is missing. Set it in .env locally, or in Streamlit Cloud under "
                  "Settings -> Secrets.",
        "pinecone": "Could not reach the Pinecone index. Check PINECONE_API_KEY and "
                    "PINECONE_INDEX_NAME, and that the index has been ingested.",
        "llm": "The LLM call failed. Check OPENAI_API_KEY and the account's quota.",
    }.get(kind or "")


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
                        raise APIError("The RAG pipeline failed.", hint=_detail_hint(event))
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


def _detail_hint(event: dict) -> Optional[str]:
    """What actually went wrong, plus what to do about it, for an engine error event."""
    parts = [event.get("message"), event.get("hint") or _hint_for(event.get("kind"))]
    return " ".join(p for p in parts if p) or None


# ---------------------------------------------------------------------------
# In-process client (no HTTP server)
# ---------------------------------------------------------------------------
class InProcessRAGClient:
    """Runs the real RAG pipeline inside the Streamlit process.

    Streamlit Community Cloud starts one process - `streamlit run streamlit_app.py` - so nothing
    is listening on 127.0.0.1:8000 and HttpRAGClient can only ever report the backend offline.
    This client drops the HTTP hop and calls src/engine.py, the same module app.py serves, so the
    deployed app does real retrieval against the real Pinecone index with the real LLM.

    src.engine is imported lazily: it pulls in langchain, langgraph and pinecone, and an
    ImportError or a missing key has to surface as a status the UI can render, not as a crash
    while Streamlit is still importing the script.
    """

    def __init__(self) -> None:
        self._engine = None
        self._load_error: Optional[str] = None

    def _load(self):
        if self._engine is None and self._load_error is None:
            try:
                from src.engine import get_engine
                self._engine = get_engine()
            except Exception as exc:
                self._load_error = f"Could not load the RAG pipeline: {exc}"
        return self._engine

    def health(self) -> SystemStatus:
        engine = self._load()
        if engine is None:
            return SystemStatus(mode="live", api=False, error=self._load_error)
        try:
            data = engine.status()
        except Exception as exc:
            return SystemStatus(mode="live", api=False, error=str(exc)[:300])
        # api=True means "the pipeline is loaded in this process"; pinecone/llm carry the live
        # checks, so a bad key shows as DEGRADED with a reason rather than a bare OFFLINE.
        errors = data.get("errors") or {}
        return SystemStatus(
            mode="live", api=True,
            pinecone=data.get("pinecone"), llm=data.get("llm"),
            error="; ".join(f"{k}: {v}" for k, v in errors.items()) or None,
        )

    def kb_info(self) -> Optional[KBInfo]:
        engine = self._load()
        if engine is None:
            return None
        try:
            return KBInfo.from_api(engine.kb_info())
        except Exception:
            return None

    def ask(self, query: str) -> RAGResponse:
        engine = self._load()
        if engine is None:
            raise APIError(self._load_error or "The RAG pipeline is unavailable.")
        from src.engine import EngineError
        try:
            return RAGResponse.from_api(engine.ask(query))
        except EngineError as exc:
            raise APIError(exc.message, hint=exc.hint or _hint_for(exc.kind)) from exc

    def stream(self, query: str) -> Iterator[dict]:
        engine = self._load()
        if engine is None:
            raise APIError(self._load_error or "The RAG pipeline is unavailable.",
                           hint="Check that the deployment installed every package in requirements.txt.")
        started = time.perf_counter()
        for event in engine.stream(query):
            if event.get("type") == "error":
                raise APIError("The RAG pipeline failed.", hint=_detail_hint(event))
            if event.get("type") == "result":
                data = event["data"]
                data.setdefault("latency", round(time.perf_counter() - started, 2))
                yield {"type": "result", "response": RAGResponse.from_api(data)}
                return
            yield event
        raise APIError("The pipeline ended before a result arrived.",
                       hint="Check the Streamlit Cloud logs for this app.")


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
        steps += [("generate", ""), ("check_grounding", "grounding 0.75"), ("finalize", "confidence 0.69")] \
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
            "confidence_score": 0.69 if in_scope else 0.0,
            "sources": [
                {"page": 7, "similarity": 0.73, "used_in_answer": in_scope},
                {"page": 19, "similarity": 0.69, "used_in_answer": in_scope},
                {"page": 42, "similarity": 0.07 if not in_scope else 0.41, "used_in_answer": False},
            ],
            "out_of_scope": not in_scope,
            # Deliberately not 1.0: claim-level grounding rarely saturates, and a preview that always
            # showed a perfect score would never exercise the partial-grounding styling.
            "grounding_score": 0.75 if in_scope else None,
            "retrieval_score": 0.71 if in_scope else None,
            "latency": 2.5,
        }
        yield {"type": "result", "response": RAGResponse.from_api(data)}


def get_client() -> RAGClient:
    """Pick a client from the environment.

        RAG_API_MODE=mock   -> MockRAGClient        (sample data, clearly labelled)
        RAG_API_URL set     -> HttpRAGClient        (a uvicorn backend is running somewhere)
        neither             -> InProcessRAGClient   (run the pipeline here)

    Defaulting to in-process rather than to 127.0.0.1:8000 is what makes the Streamlit Cloud
    deployment work: there is no second process there to connect to. Locally, put
    RAG_API_URL=http://127.0.0.1:8000 in .env to go back through FastAPI.
    """
    if os.getenv("RAG_API_MODE", "live").lower() == "mock":
        return MockRAGClient()
    api_url = (os.getenv("RAG_API_URL") or "").strip()
    if api_url:
        return HttpRAGClient(api_url)
    return InProcessRAGClient()
