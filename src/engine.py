"""The RAG pipeline as a reusable in-process engine.

Both front doors need the same three things - a health check, knowledge-base metadata, and a
streamed run of the LangGraph workflow:

    app.py (FastAPI)                -> HTTP/SSE, for local use and any API consumer
    ui/api_client.InProcessRAGClient -> direct calls, for Streamlit Community Cloud

Streamlit Cloud starts exactly one process (`streamlit run streamlit_app.py`), so there is no
uvicorn listening on 127.0.0.1:8000 for the HTTP client to reach. Putting the logic here rather
than in app.py means the deployed app runs the *same* retrieval, Pinecone index, embeddings and
LLM as the backend, instead of a second implementation that would drift.

The engine is lazy: importing this module costs nothing but imports, and nothing touches the
network until the first health check or question. That keeps a missing API key a reportable
condition rather than an import-time crash.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Iterator, Optional

from src import config
from src.graph import build_rag_graph, format_result, initial_state

NODE_NAMES = {"retrieve", "grade_documents", "rewrite_query", "generate",
              "check_grounding", "finalize", "refuse"}
HEALTH_TTL_SECONDS = 30

log = logging.getLogger("rag.engine")

# Provider errors quote the offending credential back at you ("Incorrect API key provided:
# sk-abc..."). These messages reach the browser, so scrub anything key-shaped first.
_KEY_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]{4,}|pcsk_[A-Za-z0-9_\-]{4,})")


def redact(message: str, limit: int = 300) -> str:
    return _KEY_PATTERN.sub("***", str(message))[:limit]


class EngineError(Exception):
    """A pipeline failure with a machine-readable cause, so the UI can say what broke.

    kind: "config" (a key is missing) | "pinecone" | "llm" | "pipeline"
    """

    def __init__(self, message: str, kind: str = "pipeline", hint: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.hint = hint


HINTS = {
    "config": "Add the missing key to .env locally, or to Settings -> Secrets on Streamlit Cloud.",
    "pinecone": "Check PINECONE_API_KEY and that the index exists and has been ingested.",
    "llm": "Check OPENAI_API_KEY and that the account has quota for the model.",
}


class RAGEngine:
    """Owns the compiled graph, the Pinecone handle and the PDF stats for one process."""

    def __init__(self) -> None:
        self._graph = None
        self._index = None
        self._pdf_pages: Optional[tuple[Optional[int], Optional[int]]] = None
        self._health: tuple[float, Optional[dict]] = (0.0, None)

    # --- Lazily built resources ---
    @property
    def graph(self):
        if self._graph is None:
            try:
                self._graph = build_rag_graph()
            except EnvironmentError as exc:  # config.validate(): a required key is missing
                raise EngineError(str(exc), kind="config", hint=HINTS["config"]) from exc
            except Exception as exc:
                raise EngineError(f"Could not build the RAG pipeline: {redact(exc)}",
                                  kind="pipeline") from exc
        return self._graph

    @property
    def index(self):
        if self._index is None:
            from pinecone import Pinecone  # imported here so a bad key fails as a health error
            self._index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX_NAME)
        return self._index

    @property
    def pdf_pages(self) -> tuple[Optional[int], Optional[int]]:
        """(total pages, pages carrying enough text to be chunked). (None, None) if unreadable."""
        if self._pdf_pages is None:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(config.PDF_PATH))
                with_text = sum(1 for p in reader.pages
                                if len((p.extract_text() or "").strip()) >= config.MIN_PAGE_CHARS)
                self._pdf_pages = (len(reader.pages), with_text)
            except Exception as exc:
                log.warning("Could not read %s: %s", config.PDF_PATH, exc)
                self._pdf_pages = (None, None)
        return self._pdf_pages

    def warm(self) -> None:
        """Build everything up front. Used by the FastAPI lifespan so startup fails loudly."""
        _ = self.graph, self.index, self.pdf_pages

    # --- Health and metadata ---
    def check_health(self) -> dict:
        """Real checks against Pinecone and OpenAI, cached briefly so polling stays cheap."""
        now = time.monotonic()
        cached_at, cached = self._health
        if cached and now - cached_at < HEALTH_TTL_SECONDS:
            return cached

        result = {"pinecone": False, "llm": False, "vector_count": None, "errors": {}}
        try:
            stats = self.index.describe_index_stats()
            result["pinecone"] = True
            result["vector_count"] = stats.total_vector_count
        except Exception as exc:
            result["errors"]["pinecone"] = redact(exc)
        try:
            # Free metadata call: confirms the key is valid and the model is available
            from openai import OpenAI
            OpenAI(api_key=config.OPENAI_API_KEY).models.retrieve(config.LLM_MODEL)
            result["llm"] = True
        except Exception as exc:
            result["errors"]["llm"] = redact(exc)

        self._health = (now, result)
        return result

    def status(self) -> dict:
        """The /health payload, identical whether it is served over HTTP or read in-process."""
        h = self.check_health()
        return {
            "status": "ok" if h["pinecone"] and h["llm"] else "degraded",
            "api": True,
            "pinecone": h["pinecone"],
            "llm": h["llm"],
            "errors": h["errors"],
        }

    def kb_info(self) -> dict:
        """The /kb/info payload: what the knowledge base is made of and which models serve it."""
        h = self.check_health()
        pages, pages_with_text = self.pdf_pages
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

    # --- Running a question ---
    def ask(self, question: str) -> dict:
        """One-shot run. Raises EngineError so callers can report the cause."""
        try:
            return format_result(question, self.graph.invoke(initial_state(question)))
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"RAG pipeline error: {redact(exc)}", kind="pipeline") from exc

    def stream(self, question: str) -> Iterator[dict]:
        """Yield one event per LangGraph node, then the result.

            {"type": "start", "node": ...}
            {"type": "done",  "node": ..., "detail": ...}
            {"type": "result", "data": {...}}     terminal
            {"type": "error",  "message": ..., "kind": ...}   terminal

        Errors are yielded rather than raised: the FastAPI endpoint has to put them on the SSE
        stream after the response headers have already gone out, so there is nothing left to
        raise into. The in-process client turns them back into exceptions for the UI.
        """
        try:
            graph = self.graph
        except EngineError as exc:
            yield {"type": "error", "message": exc.message, "kind": exc.kind, "hint": exc.hint}
            return

        state = initial_state(question)
        started = time.perf_counter()
        try:
            for mode, chunk in graph.stream(state, stream_mode=["updates", "debug"]):
                if mode == "debug":
                    node = _task_started(chunk)
                    if node:
                        yield {"type": "start", "node": node}
                elif mode == "updates":
                    for node, changes in chunk.items():
                        if node not in NODE_NAMES or not isinstance(changes, dict):
                            continue
                        state.update(changes)
                        yield {"type": "done", "node": node, "detail": describe(node, state)}
            result = format_result(question, state)
            result["latency"] = round(time.perf_counter() - started, 2)
            yield {"type": "result", "data": result}
        except Exception as exc:
            yield {"type": "error", "message": redact(exc), "kind": classify(exc)}


def classify(exc: Exception) -> str:
    """Best-effort guess at which dependency failed, for the UI's error message."""
    name = f"{type(exc).__module__}.{type(exc).__name__}".lower()
    text = str(exc).lower()
    if "pinecone" in name or "pinecone" in text:
        return "pinecone"
    if "openai" in name or "openai" in text or "rate limit" in text or "quota" in text:
        return "llm"
    return "pipeline"


def describe(node: str, state: dict) -> str:
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


# ---------------------------------------------------------------------------
# Process-wide singleton
#
# Streamlit re-runs the whole script on every interaction, so `get_client()` - and with it
# InProcessRAGClient.__init__ - runs again for each keystroke-sized rerun. Holding the engine at
# module level means the graph, the Pinecone connection and the PDF page scan are built once per
# process and survive those reruns; only module *imports* are cached by Python, not call results.
# ---------------------------------------------------------------------------
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
