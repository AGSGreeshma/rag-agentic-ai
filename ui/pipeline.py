"""Turns LangGraph node events from the backend into UI pipeline state.

Backend nodes                  UI steps
-----------------------------  -------------------------
retrieve                    -> Retrieve + Pinecone
grade_documents             -> Relevant context
rewrite_query               -> Relevant context (loops back to Retrieve)
generate                    -> Generate
check_grounding / finalize  -> Grounded answer
refuse                      -> Grounded answer (out of scope)

Step states: pending | active | done | success | warn | skipped
"""
from __future__ import annotations

from dataclasses import dataclass, field

STEPS = [
    ("query", "User query", "Question received"),
    ("retrieve", "Retrieve", "Embed the query with OpenAI embeddings"),
    ("pinecone", "Pinecone", "Cosine similarity search, top-k"),
    ("context", "Relevant context", "LLM grades which passages are relevant"),
    ("generate", "Generate", "Answer written only from those passages"),
    ("answer", "Grounded answer", "Groundedness check and confidence score"),
]

NODE_TO_STEPS = {
    "retrieve": ["retrieve", "pinecone"],
    "grade_documents": ["context"],
    "rewrite_query": ["context"],
    "generate": ["generate"],
    "check_grounding": ["answer"],
    "finalize": ["answer"],
    "refuse": ["answer"],
}

# Loading stages shown to the user, and the pipeline steps each one covers
STAGES = [
    ("Searching the knowledge base...", ["retrieve", "pinecone"]),
    ("Retrieving relevant passages...", ["context"]),
    ("Generating grounded response...", ["generate"]),
    ("Verifying groundedness...", ["answer"]),
]


@dataclass
class PipelineTracker:
    state: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)   # LangGraph node names, in execution order
    rewrites: int = 0
    finished: bool = False

    def __post_init__(self):
        if not self.state:
            self.state = {key: "pending" for key, _, _ in STEPS}
            self.state["query"] = "done"
            self._activate("retrieve")

    # --- helpers ---
    def _set(self, steps, value):
        for s in steps:
            self.state[s] = value

    def _activate(self, node):
        self._set(NODE_TO_STEPS.get(node, [node]), "active")

    # --- events ---
    def on_start(self, node: str) -> None:
        if node == "retrieve":
            self._set(["retrieve", "pinecone"], "active")
        elif node == "rewrite_query":
            self.state["context"] = "active"
            self.state["generate"] = "pending"
        elif node == "refuse":
            if self.state["generate"] in ("pending", "active"):
                self.state["generate"] = "skipped"
            self.state["answer"] = "active"
        else:
            self._activate(node)

    def on_done(self, node: str, detail: str = "") -> None:
        self.trace.append(node)
        if node == "retrieve":
            self._set(["retrieve", "pinecone"], "done")
            if detail:
                self.meta["pinecone"] = detail
            self.state["context"] = "active"          # grading always follows retrieval
        elif node == "grade_documents":
            self.state["context"] = "done"
            if detail:
                self.meta["context"] = detail
        elif node == "rewrite_query":
            self.rewrites += 1
            self.meta["context"] = "query rewritten, searching again"
            self._set(["retrieve", "pinecone"], "active")
        elif node == "generate":
            self.state["generate"] = "done"
            self.state["answer"] = "active"
        elif node == "check_grounding":
            if detail:
                self.meta["answer"] = detail
        elif node == "finalize":
            self.state["answer"] = "success"
        elif node == "refuse":
            if self.state["generate"] in ("pending", "active"):
                self.state["generate"] = "skipped"
            self.state["answer"] = "warn"
            self.meta["answer"] = "outside knowledge base"

    def on_event(self, event: dict) -> None:
        if event.get("type") == "start":
            self.on_start(event.get("node", ""))
        elif event.get("type") == "done":
            self.on_done(event.get("node", ""), event.get("detail", ""))

    def finish(self, out_of_scope: bool) -> None:
        self.finished = True
        for key, value in self.state.items():
            if value == "active":
                self.state[key] = "done"
        if out_of_scope:
            if self.state["generate"] == "pending":
                self.state["generate"] = "skipped"
            self.state["answer"] = "warn"
            self.meta.setdefault("answer", "outside knowledge base")
        else:
            for key in ("retrieve", "pinecone", "context", "generate"):
                if self.state[key] == "pending":
                    self.state[key] = "done"
            self.state["answer"] = "success"

    def fail(self) -> None:
        """Mark whatever was running as stopped (request failed)."""
        for key, value in self.state.items():
            if value == "active":
                self.state[key] = "error"

    # --- views ---
    def stages(self) -> list[tuple[str, str]]:
        """(message, status) for the loading indicator. status: done | active | pending"""
        out = []
        for message, steps in STAGES:
            states = [self.state[s] for s in steps]
            if any(s == "active" for s in states):
                status = "active"
            elif all(s in ("done", "success", "warn", "skipped") for s in states):
                status = "done"
            else:
                status = "pending"
            out.append((message, status))
        return out
