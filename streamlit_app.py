"""
Agentic AI Research Assistant: Streamlit frontend (chat-thread layout).

    uvicorn app:app --reload          # terminal 1: FastAPI backend
    streamlit run streamlit_app.py    # terminal 2: this UI

Set RAG_API_MODE=mock to develop the UI without a backend (sample data, clearly labelled).
The UI only talks to the backend through ui/api_client.py.
"""
import streamlit as st

from ui import components as C
from ui.api_client import APIError, get_client
from ui.pipeline import PipelineTracker

st.set_page_config(
    page_title="Agentic AI Research Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

client = get_client()


# --- Backend metadata (cached so reruns don't hit the API every time) ---
@st.cache_data(ttl=30, show_spinner=False)
def load_status():
    return client.health()


@st.cache_data(ttl=120, show_spinner=False)
def load_kb_info():
    return client.kb_info()


# --- Session state ---
st.session_state.setdefault("turns", [])      # finished turns: {query, response | error, tracker}
st.session_state.setdefault("pending", None)  # question picked from a suggestion chip


def pick(question: str) -> None:
    st.session_state.pending = question


def clear() -> None:
    st.session_state.turns = []


def answer(question: str, llm_model) -> None:
    """Stream the LangGraph run from the backend and update the live turn node by node."""
    tracker = PipelineTracker()
    live = C.LiveTurn(question)
    live.progress(tracker)
    try:
        for event in client.stream(question):
            if event["type"] == "result":
                resp = event["response"]
                tracker.finish(resp.out_of_scope)
                live.done(resp, tracker, llm_model)
                st.session_state.turns.append({"query": question, "response": resp, "tracker": tracker})
                return
            tracker.on_event(event)
            live.progress(tracker)
    except APIError as exc:
        error = ("Request failed", exc.message, exc.hint)
        load_status.clear()
    except Exception as exc:  # unexpected client-side failure
        error = ("Something went wrong", str(exc), None)
    live.error(*error, tracker)
    st.session_state.turns.append({"query": question, "error": error, "tracker": tracker})


# --- Page ---
status = load_status()
kb = load_kb_info()
llm_model = kb.llm_model if kb else None

# st.chat_input is always pinned to the bottom, wherever it's called
typed = st.chat_input("Ask anything about Agentic AI...", max_chars=1000)
question = (typed or st.session_state.pending or "").strip()
st.session_state.pending = None

C.inject_theme()
C.sidebar(kb, status, on_clear=clear)
C.header(status)

if not st.session_state.turns and not question:
    C.welcome(status, kb, on_pick=pick)

C.history(st.session_state.turns, llm_model)

if question:
    answer(question, llm_model)

C.footer()
