# Agentic AI Research Assistant: UI

Light pastel theme with Instrument Serif Italic display type (loaded from Google Fonts). Chat-thread layout: questions and answers stack as a conversation, the input is pinned to the bottom,
and every answer carries a groundedness chip plus two collapsible drawers:
**Sources** (the retrieved passages with page and match score) and **How this answer was generated**
(the live LangGraph pipeline, execution trace and any rewritten queries).

## Run it

Two terminals, both with the venv active, from the project root:

```bash
uvicorn app:app --reload          # 1. FastAPI backend  -> http://127.0.0.1:8000/docs
streamlit run streamlit_app.py    # 2. Streamlit UI     -> http://localhost:8501
```

To design or demo the UI without the backend (clearly labelled sample data):

```bash
# macOS / Linux
RAG_API_MODE=mock streamlit run streamlit_app.py
# Windows PowerShell
$env:RAG_API_MODE="mock"; streamlit run streamlit_app.py; Remove-Item Env:RAG_API_MODE
```

Point the UI at a different backend with `RAG_API_URL` (default `http://127.0.0.1:8000`).

## Frontend architecture

```
streamlit_app.py      Page flow and session state only
ui/
  api_client.py       API service layer: HttpRAGClient (FastAPI) and MockRAGClient, same interface
  models.py           RAGResponse, Source, KBInfo, SystemStatus: normalises any backend response
  pipeline.py         PipelineTracker: turns LangGraph node events into UI pipeline steps
  html.py             Pure HTML builders for every component (testable without Streamlit)
  components.py       Streamlit components: header, welcome, history, live turn, sidebar, footer
  theme.py            Design tokens and CSS
.streamlit/config.toml  Dark base theme
```

## Backend endpoints used by the UI

| Endpoint | Purpose |
|---|---|
| `GET /chat/stream?query=...` | Server-Sent Events: one event per LangGraph node, then the final result. Drives the live pipeline view. |
| `POST /chat` | Assignment response format (`query`, `final_answer`, `retrieved_context_chunks`, `confidence_score`) plus page/similarity metadata. The UI falls back to it if streaming is unavailable. |
| `GET /health` | Real checks: Pinecone `describe_index_stats`, OpenAI model lookup. Cached for 30 s. |
| `GET /kb/info` | Page count from the PDF, chunk count from Pinecone, model names from config. |

Nothing in the sidebar or header is hard-coded: if the backend can't confirm a value, the UI shows "—" or "Unavailable".
