"""Pure HTML builders for every UI component (no Streamlit imports).

Keeping these as plain functions that return strings makes them easy to test and
reuse; ui/components.py is the thin layer that hands them to Streamlit.
Note: output must contain no blank lines or indented lines, because Streamlit's
markdown parser would otherwise treat parts of the HTML as text or code.
"""
from __future__ import annotations

import html
import json
import re
from typing import Optional

from ui.models import KBInfo, RAGResponse, Source, SystemStatus
from ui.pipeline import STEPS, PipelineTracker

OUT_OF_SCOPE_MESSAGE = (
    "I couldn't find sufficient information about this question in the provided Agentic AI eBook."
)


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _join(*parts: str) -> str:
    return "".join(p.strip() for p in parts if p)


def _fmt(value, fmt="{}", empty="—") -> str:
    return empty if value is None else fmt.format(value)


# ---------------------------------------------------------------------------
# Icons (inline SVG)
# ---------------------------------------------------------------------------
NETWORK_ICON = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M6 7l6-3 6 3M6 7v10M18 7v10M6 17l6 3 6-3M12 4v7M6 7l6 4 6-4M12 11v9" '
    'stroke="#9A92AB" stroke-width="1.2" stroke-linejoin="round"/>'
    '<circle cx="12" cy="4" r="1.7" fill="#8E7BE6"/><circle cx="6" cy="7" r="1.5" fill="#5DB5A8"/>'
    '<circle cx="18" cy="7" r="1.5" fill="#5DB5A8"/><circle cx="12" cy="11" r="1.9" fill="#8E7BE6"/>'
    '<circle cx="6" cy="17" r="1.5" fill="#EFA3BC"/><circle cx="18" cy="17" r="1.5" fill="#EFA3BC"/>'
    '<circle cx="12" cy="20" r="1.5" fill="#F2B98F"/></svg>'
)


def _hero_art() -> str:
    """Abstract vector-space network: thin violet/cyan edges between points. Deterministic."""
    pts = [(40, 150), (78, 92), (118, 160), (150, 58), (176, 118), (214, 172), (236, 84),
           (272, 132), (298, 52), (318, 170), (346, 104), (96, 34), (254, 26), (196, 40)]
    violet_edges = [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8), (6, 7), (7, 10), (3, 13), (13, 12), (12, 8), (1, 11)]
    cyan_edges = [(1, 2), (2, 4), (4, 5), (5, 7), (7, 9), (9, 10), (0, 2), (11, 3)]
    hubs = {3, 4, 6, 7}
    lines = []
    for a, b in violet_edges:
        (x1, y1), (x2, y2) = pts[a], pts[b]
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#8E7BE6" stroke-opacity="0.55" stroke-width="1"/>')
    for a, b in cyan_edges:
        (x1, y1), (x2, y2) = pts[a], pts[b]
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#5DB5A8" stroke-opacity="0.5" stroke-width="0.9"/>')
    dots = []
    for i, (x, y) in enumerate(pts):
        color = "#8E7BE6" if i in hubs else ("#5DB5A8", "#EFA3BC", "#F2B98F")[i % 3]
        r = 3.2 if i in hubs else 2
        if i in hubs:
            dots.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{color}" fill-opacity="0.18"/>')
        dots.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" fill-opacity="0.9"/>')
    # faint concentric arcs suggesting an embedding space
    rings = ('<circle cx="190" cy="104" r="70" fill="none" stroke="#E3DBF2" stroke-width="1"/>'
             '<circle cx="190" cy="104" r="118" fill="none" stroke="#E3DBF2" stroke-width="0.8" stroke-dasharray="2 5"/>')
    return (f'<svg viewBox="0 0 380 200" preserveAspectRatio="xMidYMid slice" aria-hidden="true">'
            f'{rings}{"".join(lines)}{"".join(dots)}</svg>')


# ---------------------------------------------------------------------------
# Header, hero, footer
# ---------------------------------------------------------------------------
def header_html(status: SystemStatus) -> str:
    if status.mode == "mock":
        dot, label = "warn", "MOCK MODE"
    elif not status.api:
        dot, label = "off", "RAG OFFLINE"
    elif status.online:
        dot, label = "ok", "RAG ONLINE"
    else:
        dot, label = "warn", "RAG DEGRADED"
    banner = ""
    if status.mode == "mock":
        banner = ('<div class="ara-banner">Mock mode: responses are sample data from the mock client, '
                  'not from the RAG backend. Unset RAG_API_MODE to connect to FastAPI.</div>')
    elif status.error:
        # Say which dependency failed. Without this the UI can only show OFFLINE/DEGRADED, which
        # is the same picture for a missing API key, an unreachable index and a bad model name.
        banner = f'<div class="ara-banner">{_e(status.error)}</div>'
    return _join(
        '<div class="ara-header">',
        '<div class="ara-brand">',
        f'<div class="ara-logo">{NETWORK_ICON}</div>',
        '<div><div class="ara-brand-name">Agentic AI <span>Research Assistant</span></div>',
        '<div class="ara-brand-sub">Grounded answers from the Agentic AI eBook</div></div>',
        '</div>',
        f'<div class="ara-status"><span class="ara-dot {dot}"></span>{label}</div>',
        '</div>',
        banner,
    )


def footer_html() -> str:
    return _join(
        '<div class="ara-footer"><span>Agentic AI Research Assistant</span>',
        '<span>Powered by LangGraph • Pinecone • RAG</span></div>',
    )


def welcome_html(status: SystemStatus, kb: Optional[KBInfo]) -> str:
    """Empty state: shown before the first question."""
    if status.mode == "mock":
        ready = '<span class="ara-dot warn"></span>Mock mode: sample data only.'
    elif status.online and kb and kb.chunks:
        ready = ('<span class="ara-dot ok"></span>Your Agentic AI knowledge base is ready. '
                 'Ask a question to explore the eBook.')
    elif not status.api:
        ready = '<span class="ara-dot off"></span>The RAG API is offline. Start the backend to ask questions.'
    else:
        ready = '<span class="ara-dot warn"></span>Some services are unavailable. See System status in the sidebar.'
    return _join(
        '<div class="ara-welcome">',
        f'<div class="ara-welcome-art">{_hero_art()}</div>',
        '<h1>Ask the <em>eBook.</em></h1>',
        '<p>Explore Agentic AI through answers grounded exclusively in the provided knowledge base.</p>',
        f'<div class="ara-welcome-ready">{ready}</div>',
        '</div>',
    )


# ---------------------------------------------------------------------------
# Pipeline ("Steps" drawer)
# ---------------------------------------------------------------------------
def _stages_html(tracker: PipelineTracker) -> str:
    rows = []
    for message, status in tracker.stages():
        mark = "✓" if status == "done" else ""
        rows.append(f'<div class="ara-stage {status}"><span class="ara-stage-mark">{mark}</span>{_e(message)}</div>')
    return "".join(rows)


def pipeline_html(tracker: PipelineTracker, response: Optional[RAGResponse] = None) -> str:
    meta = dict(tracker.meta)
    meta.setdefault("query", "")
    if response and not response.out_of_scope and response.grounding_score is not None:
        meta["answer"] = f"grounding {response.grounding_score:.2f}"
    rows = []
    for i, (key, title, sub) in enumerate(STEPS):
        state = tracker.state.get(key, "pending")
        last = " last" if i == len(STEPS) - 1 else ""
        m = {"skipped": "skipped", "error": "stopped"}.get(state, meta.get(key, ""))
        rows.append(_join(
            f'<div class="ara-step {state}{last}">',
            '<div class="ara-rail"><span class="ara-node"></span><span class="ara-line"></span></div>',
            f'<div class="ara-step-body"><div class="ara-step-title">{_e(title)}</div>',
            f'<div class="ara-step-sub">{_e(sub)}</div></div>',
            f'<div class="ara-step-meta">{_e(m)}</div>',
            '</div>',
        ))
    foot = ""
    if tracker.trace:
        trace = " → ".join(f"<b>{_e(n)}</b>" for n in tracker.trace)
        rewrites = ""
        if response and response.rewritten_queries:
            rewrites = _join('<div class="ara-rewrites"><div class="ara-label" style="margin:0 0 2px">Rewritten queries</div>',
                             *[f'<div class="q">{_e(q)}</div>' for q in response.rewritten_queries], "</div>")
        foot = _join('<div class="ara-pipe-foot">',
                     f'<div class="ara-trace">LangGraph trace: {trace}</div>', rewrites, '</div>')
    return _join('<div class="ara-pipe">', *rows, foot, '</div>')


# ---------------------------------------------------------------------------
# Answer text
# ---------------------------------------------------------------------------
_CITE = re.compile(r"\(?\bp\.\s?(\d+)\)?([.,;:]?)")


def _inline(text: str) -> str:
    text = _e(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # keep trailing punctuation on the same line as the citation chip
    return _CITE.sub(lambda m: f'<span class="ara-nowrap"><span class="ara-cite">p.{m.group(1)}</span>{m.group(2)}</span>', text)


def markdown_to_html(src: str) -> str:
    """Minimal, safe Markdown: paragraphs, bullet/numbered lists, bold, page citations."""
    out, para, in_list = [], [], False

    def flush():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    for raw in src.splitlines():
        line = raw.strip()
        item = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if item:
            flush()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(item.group(1))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line:
            para.append(line)
        else:
            flush()
    flush()
    if in_list:
        out.append("</ul>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Sources ("Sources" drawer)
# ---------------------------------------------------------------------------
def _source_card(src: Source, preview_chars: int = 230) -> str:
    text = " ".join(src.text.split())
    preview = text if len(text) <= preview_chars else text[:preview_chars].rsplit(" ", 1)[0] + "…"
    page = f"PAGE {src.page}" if src.page is not None else "PASSAGE"
    match = ""
    if src.similarity is not None:
        width = round(max(0.0, min(1.0, src.similarity)) * 100)
        match = (f'<div class="ara-src-match" title="Cosine similarity between the query and this passage">'
                 f'MATCH {src.similarity:.2f}<span class="bar"><i style="width:{width}%"></i></span></div>')
    cls = "ara-src" if src.used_in_answer else "ara-src unused"
    return _join(
        f'<details class="{cls}"><summary>',
        f'<div class="ara-src-top"><span class="ara-src-page">{page}</span>{match}</div>',
        f'<div class="ara-src-preview">“{_e(preview)}”</div>',
        '<span class="ara-src-more"><span class="more">View full passage →</span><span class="less">Hide passage ↑</span></span>',
        '</summary>',
        f'<div class="ara-src-full">{_e(src.text.strip())}</div>',
        '</details>',
    )


def sources_html(resp: RAGResponse) -> str:
    if resp.out_of_scope:
        sub = "Closest passages found. None were relevant enough to answer from."
        cards, hidden = resp.sources, []
    else:
        sub = "Source passages used to generate this answer"
        cards, hidden = resp.used_sources, resp.other_sources
    extra = ""
    if hidden:
        n = len(hidden)
        extra = _join(
            f'<details class="ara-others"><summary>Show {n} other retrieved passage{"s" if n != 1 else ""} '
            '(not used in the answer)</summary>',
            "".join(_source_card(s) for s in hidden), "</details>")
    return _join(f'<div class="ara-drawer-sub">{sub}</div>', "".join(_source_card(s) for s in cards), extra)


# ---------------------------------------------------------------------------
# Chat turns
# ---------------------------------------------------------------------------
AVATAR = ('<div class="ara-avatar"><svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">'
          '<path d="M8 1l7 7-7 7-7-7z" fill="#8E7BE6"/><path d="M8 5l3 3-3 3-3-3z" fill="#FFFFFF"/></svg></div>')


def user_turn_html(query: str) -> str:
    return f'<div class="ara-turn user"><div class="ara-bubble">{_e(query)}</div></div>'


def _ai_turn(body: str, meta: str = "") -> str:
    return _join(
        '<div class="ara-turn ai">', AVATAR,
        '<div class="ara-ai">',
        f'<div class="ara-ai-head"><b>Research Assistant</b>{meta}</div>',
        body,
        '</div></div>',
    )


def thinking_html(tracker: PipelineTracker) -> str:
    return _ai_turn(f'<div class="ara-thinking">{_stages_html(tracker)}</div>', "<span>working</span>")


def _confidence_chip(resp: RAGResponse) -> str:
    if resp.out_of_scope:
        return ('<span class="ara-chip conf oos" title="Outside the knowledge base: no passage in the eBook was relevant enough to answer from.">'
                '<span class="ara-dot warn"></span>Confidence 0.00</span>')
    value = resp.confidence
    level = "high" if value >= 0.75 else "mid" if value >= 0.5 else "low"
    parts = []
    if resp.retrieval_score is not None:
        parts.append(f"retrieval similarity {resp.retrieval_score:.2f}")
    if resp.grounding_score is not None:
        parts.append(f"grounding score {resp.grounding_score:.2f}")
    tip = ("Groundedness confidence: how well the answer is supported by the retrieved passages, "
           "not a measure of factual certainty." + (f" ({', '.join(parts)})" if parts else ""))
    dot = {"high": "ok", "mid": "cyan", "low": "warn"}[level]
    return (f'<span class="ara-chip conf {level}" title="{_e(tip)}">'
            f'<span class="ara-dot {dot}"></span>Confidence {max(0.0, min(1.0, value)):.2f}</span>')


def _drawer(label: str, body: str, icon: str = "") -> str:
    return _join(
        '<details class="ara-drawer">',
        f'<summary class="ara-chip">{icon}{_e(label)}<span class="caret">▾</span></summary>',
        f'<div class="ara-drawer-body">{body}</div>',
        '</details>',
    )


def _out_of_scope_block(resp: RAGResponse) -> str:
    best = max((s.similarity for s in resp.sources if s.similarity is not None), default=None)
    return _join(
        '<div class="ara-oos">',
        '<div class="ara-oos-head"><div class="ara-oos-icon">!</div>',
        '<div class="ara-oos-title">Outside Knowledge Base</div></div>',
        f'<p>{OUT_OF_SCOPE_MESSAGE}</p>',
        '<div class="ara-oos-grid">',
        '<div><span>Retrieved context</span><b>Insufficient</b></div>',
        '<div><span>Groundedness</span><b>Low</b></div>',
        f'<div><span>Best passage match</span><b>{_fmt(best, "{:.2f}")}</b></div>',
        '</div>',
        '<p class="ara-oos-note">The assistant does not fall back to general model knowledge. '
        'Try rephrasing the question around a topic the eBook covers.</p>',
        '</div>',
    )


REQUIRED_FIELDS = ("query", "final_answer", "retrieved_context_chunks", "confidence_score")


def json_payload_html(resp: RAGResponse) -> str:
    """The exact assignment-format JSON for this answer, as returned by POST /chat."""
    src = resp.raw or {
        "query": resp.query, "final_answer": resp.answer,
        "retrieved_context_chunks": [s.text for s in resp.sources], "confidence_score": resp.confidence,
    }
    payload = {k: src.get(k) for k in REQUIRED_FIELDS}
    # newlines as entities so Streamlit's markdown parser leaves the <pre> block intact
    body = _e(json.dumps(payload, indent=2, ensure_ascii=False)).replace("\n", "&#10;")
    return _join(
        '<div class="ara-drawer-sub">Structured response in the assignment format '
        '(<code>POST /chat</code> also returns page, similarity and grounding fields)</div>',
        f'<pre class="ara-json">{body}</pre>',
    )


def assistant_turn_html(resp: RAGResponse, tracker: PipelineTracker, llm_model: Optional[str] = None) -> str:
    meta_bits = []
    if llm_model and not resp.out_of_scope:
        meta_bits.append(_e(llm_model))
    if resp.latency is not None:
        meta_bits.append(f"{resp.latency:.1f}s")
    if resp.rewritten_queries:
        meta_bits.append("query rewritten")
    meta = f"<span>{' · '.join(meta_bits)}</span>" if meta_bits else ""

    body = _out_of_scope_block(resp) if resp.out_of_scope else \
        f'<div class="ara-answer-body">{markdown_to_html(resp.answer)}</div>'

    if resp.out_of_scope:
        src_label = f"{len(resp.sources)} closest passages"
    else:
        n = len(resp.used_sources)
        src_label = f"{n} source{'s' if n != 1 else ''}"
    src_icon = '<span class="ara-chip-ico cyan">❝</span>'
    step_icon = '<span class="ara-chip-ico violet">⋮</span>'
    actions = _join(
        '<div class="ara-actions">',
        _confidence_chip(resp),
        _drawer(src_label, sources_html(resp), src_icon) if resp.sources else "",
        _drawer("How this answer was generated", pipeline_html(tracker, resp), step_icon),
        _drawer("JSON response", json_payload_html(resp), '<span class="ara-chip-ico violet">{ }</span>'),
        '</div>',
    )
    return _ai_turn(body + actions, meta)


def error_html(title: str, message: str, hint: Optional[str] = None) -> str:
    return _join(
        '<div class="ara-error">',
        f'<div class="ara-error-title">{_e(title)}</div>',
        f'<p>{_e(message)}</p>',
        f'<code>{_e(hint)}</code>' if hint else "",
        '</div>',
    )


def assistant_error_html(title: str, message: str, hint: Optional[str], tracker: PipelineTracker) -> str:
    steps = _drawer("Where it stopped", pipeline_html(tracker), '<span class="ara-chip-ico violet">⋮</span>')
    return _ai_turn(error_html(title, message, hint) + f'<div class="ara-actions">{steps}</div>')


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _kv(label: str, value: str, mono: bool = False, stack: bool = False) -> str:
    return f'<div class="ara-kv{" stack" if stack else ""}"><span>{_e(label)}</span><b class="{"mono" if mono else ""}">{_e(value)}</b></div>'


def sidebar_kb_html(kb: Optional[KBInfo], status: SystemStatus) -> str:
    kb = kb or KBInfo()
    if kb.chunks:
        state = '<span class="ara-dot ok"></span>Indexed'
    elif kb.chunks == 0:
        state = '<span class="ara-dot warn"></span>Index is empty. Run ingestion'
    else:
        state = '<span class="ara-dot idle"></span>Index status unknown'
    pages = _fmt(kb.pages)
    if kb.pages is not None and kb.pages_with_text is not None:
        pages = f"{kb.pages} ({kb.pages_with_text} with text)"
    return _join(
        '<div class="ara-side-section">',
        '<div class="ara-label">Knowledge base</div>',
        '<div class="ara-kb-doc">📘 Agentic AI eBook</div>',
        f'<div class="ara-kb-file">{_e(kb.document or "—")}</div>',
        f'<div class="ara-kb-state">{state}</div>',
        _kv("Pages", pages),
        _kv("Chunks", _fmt(kb.chunks)),
        _kv("Vector DB", kb.vector_db or "Pinecone"),
        _kv("Index", kb.index_name or "—", mono=True),
        _kv("Embeddings · OpenAI", kb.embedding_model or "—", mono=True, stack=True),
        _kv("LLM", kb.llm_model or "—", mono=True),
        '</div>',
    )


def sidebar_pipeline_html(kb: Optional[KBInfo], status: SystemStatus) -> str:
    indexed = bool(kb and kb.chunks)
    items = [
        ("Document ingestion", indexed),
        ("Chunking", indexed),
        ("Embedding", indexed),
        ("Vector retrieval", indexed and bool(status.pinecone)),
        ("Grounded generation", bool(status.llm)),
    ]
    rows = [f'<div class="ara-check {"ok" if ok else "pending"}"><i>{"✓" if ok else ""}</i>{_e(name)}</div>'
            for name, ok in items]
    return _join('<div class="ara-side-section"><div class="ara-label">RAG pipeline</div>', *rows, '</div>')


def sidebar_status_html(status: SystemStatus) -> str:
    def row(name: str, ok: Optional[bool]) -> str:
        if status.mode == "mock":
            dot, text = "warn", "Mock"
        elif ok is None:
            dot, text = "idle", "—"
        elif ok:
            dot, text = "ok", "Connected"
        else:
            dot, text = "off", "Unavailable"
        return f'<div class="ara-sys"><span><span class="ara-dot {dot}"></span>{_e(name)}</span><em>{text}</em></div>'

    api_ok = status.api if status.mode == "live" else None
    pinecone = status.pinecone if status.api else None
    llm = status.llm if status.api else None
    return _join(
        '<div class="ara-side-section"><div class="ara-label">System status</div>',
        row("Pinecone", pinecone), row("LLM", llm), row("API", api_ok),
        '</div>',
    )
