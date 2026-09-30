"""Streamlit components for the chat-thread layout.

HTML lives in ui/html.py (pure functions); this module only wires it into
Streamlit and handles the interactive widgets (suggestion chips, clear button).
"""
from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from ui import html as H
from ui.models import KBInfo, RAGResponse, SystemStatus
from ui.pipeline import PipelineTracker
from ui.theme import CSS

SUGGESTIONS = [
    "What is Agentic AI?",
    "How are agentic systems architected?",
    "What are the major use cases?",
    "How does Agentic AI differ from traditional chatbots?",
    "What challenges are discussed?",
]


def _html(markup: str, target=None) -> None:
    (target or st).markdown(markup, unsafe_allow_html=True)


def inject_theme() -> None:
    _html(CSS)


def header(status: SystemStatus) -> None:
    _html(H.header_html(status))


def footer() -> None:
    _html(H.footer_html())


def sidebar(kb: Optional[KBInfo], status: SystemStatus, on_clear: Callable[[], None]) -> None:
    with st.sidebar:
        _html(H.sidebar_kb_html(kb, status))
        _html(H.sidebar_pipeline_html(kb, status))
        _html(H.sidebar_status_html(status))
        st.button("Clear conversation", on_click=on_clear, key="clear_btn")


def welcome(status: SystemStatus, kb: Optional[KBInfo], on_pick: Callable[[str], None]) -> None:
    """Empty state with clickable example questions."""
    _html(H.welcome_html(status, kb))
    _html('<div class="ara-chips-label">Try one of these</div>')
    with st.container(key="chips"):
        cols = st.columns(len(SUGGESTIONS))
        for i, (col, q) in enumerate(zip(cols, SUGGESTIONS)):
            with col:
                st.button(q, key=f"chip_{i}", on_click=on_pick, args=(q,))


def history(turns: list[dict], llm_model: Optional[str]) -> None:
    """Re-render every finished turn of the conversation."""
    for turn in turns:
        _html(H.user_turn_html(turn["query"]))
        if turn.get("response") is not None:
            _html(H.assistant_turn_html(turn["response"], turn["tracker"], llm_model))
        else:
            title, message, hint = turn["error"]
            _html(H.assistant_error_html(title, message, hint, turn["tracker"]))


class LiveTurn:
    """The turn currently being answered: the user's bubble plus an assistant
    placeholder that updates as each LangGraph node reports progress."""

    def __init__(self, query: str):
        _html(H.user_turn_html(query))
        self.slot = st.empty()

    def progress(self, tracker: PipelineTracker) -> None:
        _html(H.thinking_html(tracker), self.slot)

    def done(self, resp: RAGResponse, tracker: PipelineTracker, llm_model: Optional[str]) -> None:
        _html(H.assistant_turn_html(resp, tracker, llm_model), self.slot)

    def error(self, title: str, message: str, hint: Optional[str], tracker: PipelineTracker) -> None:
        tracker.fail()
        _html(H.assistant_error_html(title, message, hint, tracker), self.slot)
