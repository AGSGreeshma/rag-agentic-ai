"""
LangGraph RAG workflow:
retrieve -> grade_documents -> (rewrite_query -> retrieve) -> generate -> check_grounding -> finalize / refuse
"""
from typing import Literal, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config

REFUSAL = "I cannot answer this based on the provided document."

RELEVANCE_PROMPT = """You are grading passages retrieved from an eBook about Agentic AI.

Question: {question}

Passages:
{chunks}

Return the numbers of the passages that contain information that helps answer the
question, even partially. Return an empty list if none do. A passage that only shares
a keyword but does not help answer the question is NOT relevant."""

REWRITE_PROMPT = """You write search queries for a vector database built from an eBook about Agentic AI.
Almost every passage in the eBook mentions "Agentic AI", so that phrase is useless for search.

Write 3 different search queries for the question below:
1. A short list of the topic's key terms plus synonyms
   (e.g. "challenges, risks, barriers, limitations, concerns").
2. A phrase worded like a section heading in a business eBook
   (e.g. "Challenges and mitigation strategies of multi-agent systems").
3. A phrase from the angle of an organization implementing it
   (e.g. "challenges organizations face when implementing and adopting AI agents").

Do NOT use the words "Agentic AI", "eBook", "document" or "text" in the queries.

Question: {question}"""

ANSWER_PROMPT = """You answer questions using ONLY the context below, taken from an eBook about Agentic AI.

Rules:
- Use only information in the context. Never use outside knowledge.
- If the context contains only part of the answer, answer with what it does contain.
- Only if the context contains nothing relevant to the question, reply exactly: "{refusal}"
- Cite pages inline, like (p. 12), using ONLY the page numbers shown in the
  [Passage N | page X] headers. Ignore any page numbers that appear inside the text itself.
- Be concise: a short paragraph or a few bullet points.

Context:
{context}

Question: {question}
Answer:"""

GROUNDING_PROMPT = """You are auditing an AI-generated answer for hallucinations.

Context:
{context}

Answer:
{answer}

Break the answer into its individual factual claims - each specific assertion it makes about
Agentic AI. Ignore framing sentences that assert nothing ("Here are the key points:") and ignore
the page citations themselves.

Grade every claim independently against the context above:
- "supported"   = the context states this directly; rephrasing is fine as long as the meaning matches
- "partial"     = the context touches this but the answer overstates, generalises, or adds detail
- "unsupported" = the context does not state this at all, even if it is true in the real world

Judge each claim only against the context, never against your own knowledge. An answer that is
factually correct but not present in the context is "unsupported". Do not assume the answer is
right because it sounds plausible - list unsupported claims when you find them."""


# ---------------------------------------------------------------------------
# State and structured-output schemas
# ---------------------------------------------------------------------------
class Chunk(TypedDict):
    text: str
    page: int
    similarity: float
    relevant: bool


class AgentState(TypedDict):
    question: str
    search_queries: list[str]
    rewrites: int
    chunks: list[Chunk]
    answer: str
    refused: bool
    grounding_score: float
    confidence_score: float
    attempts: int


class RelevanceGrade(BaseModel):
    relevant_ids: list[int] = Field(description="Numbers of the passages that help answer the question")


class RewrittenQueries(BaseModel):
    queries: list[str] = Field(description="Three alternative search queries")


class ClaimVerdict(BaseModel):
    claim: str = Field(description="One factual claim made by the answer, quoted or closely paraphrased")
    verdict: Literal["supported", "partial", "unsupported"] = Field(
        description="Whether the context supports this claim"
    )


class GroundingGrade(BaseModel):
    claims: list[ClaimVerdict] = Field(description="Every factual claim in the answer, each graded")


CLAIM_WEIGHTS = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


def grounding_from_claims(claims: list[ClaimVerdict]) -> float:
    """Groundedness as the weighted share of the answer's claims the context supports.

    Scoring the claims in Python rather than asking the LLM for a single float: a model asked to
    self-rate an answer it just wrote returns 1.0 almost every time, but it will mark individual
    claims unsupported when made to enumerate them. No claims extracted means we learned nothing
    about the answer, so it fails closed (one stricter retry, then refuse)."""
    if not claims:
        return 0.0
    return sum(CLAIM_WEIGHTS[c.verdict] for c in claims) / len(claims)


def is_refusal(answer: str) -> bool:
    """Text match for the refusal message. Called in exactly one place - `generate` - which records
    the result as `state["refused"]`; everything downstream routes on that flag rather than
    re-inspecting free text, so a paraphrased refusal can only be misread once."""
    return REFUSAL.lower().rstrip(".") in answer.lower()


# ---------------------------------------------------------------------------
# Pure helpers
#
# These hold the pipeline's decision logic and its arithmetic. They take only a state dict and
# live at module level, so tests can drive them against hand-built states with no API keys, no
# network and no cost. `build_rag_graph` wires the two routers in as the graph's conditional edges.
# ---------------------------------------------------------------------------
def format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[Passage {i} | page {c['page']}]\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )


def relevant_chunks(state: AgentState) -> list[Chunk]:
    return [c for c in state["chunks"] if c["relevant"]]


def retrieval_score(state: AgentState) -> float | None:
    """Mean similarity of the chunks that made it into the answer; None when there are none."""
    relevant = relevant_chunks(state)
    if not relevant:
        return None
    return round(sum(c["similarity"] for c in relevant) / len(relevant), 4)


def compute_confidence(state: AgentState) -> float:
    """confidence = RETRIEVAL_WEIGHT * retrieval + (1 - RETRIEVAL_WEIGHT) * grounding."""
    retrieval = retrieval_score(state)
    if retrieval is None:
        return 0.0
    confidence = (config.RETRIEVAL_WEIGHT * retrieval
                  + (1 - config.RETRIEVAL_WEIGHT) * state["grounding_score"])
    return round(confidence, 2)


def route_after_grading(state: AgentState) -> str:
    relevant = relevant_chunks(state)
    on_topic = any(c["similarity"] >= config.MIN_SIMILARITY for c in state["chunks"])
    enough = len(relevant) >= config.MIN_RELEVANT_CHUNKS
    out_of_rewrites = state["rewrites"] >= config.MAX_QUERY_REWRITES
    if enough or not on_topic or out_of_rewrites:
        return "generate" if relevant else "refuse"
    return "rewrite_query"


def route_after_grounding(state: AgentState) -> str:
    if state["refused"]:
        return "refuse"
    if state["grounding_score"] >= config.GROUNDING_THRESHOLD:
        return "finalize"
    if state["attempts"] < config.MAX_GENERATION_ATTEMPTS:
        return "generate"
    return "refuse"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_rag_graph():
    config.validate()
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    relevance_grader = llm.with_structured_output(RelevanceGrade)
    query_rewriter = llm.with_structured_output(RewrittenQueries)
    grounding_grader = llm.with_structured_output(GroundingGrade)

    # --- Nodes ---
    def retrieve(state: AgentState):
        queries = state["search_queries"] or [state["question"]]
        # Keep chunks already judged relevant on a previous pass
        merged: dict[str, Chunk] = {c["text"]: c for c in relevant_chunks(state)}
        # Take the top-k of EACH query (scores from different queries aren't comparable,
        # so we merge rather than re-rank them against each other)
        for query in queries:
            for doc, score in vector_store.similarity_search_with_score(query, k=config.TOP_K):
                text = doc.page_content
                if text in merged:
                    merged[text]["similarity"] = max(merged[text]["similarity"], round(float(score), 4))
                else:
                    merged[text] = {
                        "text": text,
                        "page": int(doc.metadata.get("page", 0)),
                        "similarity": round(float(score), 4),
                        "relevant": False,
                    }
        return {"chunks": list(merged.values())}

    def grade_documents(state: AgentState):
        candidates = [c for c in state["chunks"] if c["similarity"] >= config.MIN_SIMILARITY]
        if not candidates:
            return {"chunks": state["chunks"]}
        grade = relevance_grader.invoke(
            RELEVANCE_PROMPT.format(question=state["question"], chunks=format_chunks(candidates))
        )
        keep = {candidates[i - 1]["text"] for i in grade.relevant_ids if 1 <= i <= len(candidates)}
        return {"chunks": [{**c, "relevant": c["text"] in keep} for c in state["chunks"]]}

    def rewrite_query(state: AgentState):
        result = query_rewriter.invoke(REWRITE_PROMPT.format(question=state["question"]))
        queries = [q.strip() for q in result.queries if q.strip()][:3]
        return {"search_queries": queries, "rewrites": state["rewrites"] + 1}

    def generate(state: AgentState):
        prompt = ANSWER_PROMPT.format(
            refusal=REFUSAL,
            context=format_chunks(relevant_chunks(state)),
            question=state["question"],
        )
        if state["attempts"] > 0:
            prompt += ("\n\nNote: your previous answer contained claims not supported by the context. "
                       "Only state what the context explicitly says.")
        answer = llm.invoke(prompt).content.strip()
        return {"answer": answer, "refused": is_refusal(answer), "attempts": state["attempts"] + 1}

    def check_grounding(state: AgentState):
        if state["refused"]:
            return {"grounding_score": 0.0}
        grade = grounding_grader.invoke(
            GROUNDING_PROMPT.format(context=format_chunks(relevant_chunks(state)), answer=state["answer"])
        )
        return {"grounding_score": grounding_from_claims(grade.claims)}

    def finalize(state: AgentState):
        return {"confidence_score": compute_confidence(state)}

    def refuse(state: AgentState):
        return {"answer": REFUSAL, "refused": True, "confidence_score": 0.0}

    # --- Wiring ---
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_grounding", check_grounding)
    workflow.add_node("finalize", finalize)
    workflow.add_node("refuse", refuse)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "refuse": "refuse"},
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "check_grounding")
    workflow.add_conditional_edges(
        "check_grounding", route_after_grounding,
        {"finalize": "finalize", "generate": "generate", "refuse": "refuse"},
    )
    workflow.add_edge("finalize", END)
    workflow.add_edge("refuse", END)
    return workflow.compile()


# ---------------------------------------------------------------------------
# Public helpers: starting state, response shaping, one-shot query
# ---------------------------------------------------------------------------
def initial_state(question: str) -> dict:
    """The starting state for one question."""
    return {
        "question": question, "search_queries": [], "rewrites": 0,
        "chunks": [], "answer": "", "refused": False,
        "grounding_score": 0.0, "confidence_score": 0.0, "attempts": 0,
    }


def format_result(question: str, state: dict) -> dict:
    """Shape the final graph state into the API response.
    The first four keys are the required assignment format; the rest add transparency."""
    refused = state["refused"]
    retrieval = retrieval_score(state)
    return {
        "query": question,
        "final_answer": state["answer"],
        "retrieved_context_chunks": [c["text"] for c in state["chunks"]],
        "confidence_score": state["confidence_score"],
        "sources": [
            {"page": c["page"], "similarity": c["similarity"], "used_in_answer": c["relevant"]}
            for c in state["chunks"]
        ],
        "rewritten_queries": state["search_queries"],
        "out_of_scope": refused,
        "grounding_score": None if refused else round(state["grounding_score"], 2),
        "retrieval_score": None if refused else retrieval,
    }


def run_query(graph, question: str) -> dict:
    """Invoke the graph and shape the output into the required response format."""
    return format_result(question, graph.invoke(initial_state(question)))


if __name__ == "__main__":
    import json
    import sys

    question = " ".join(sys.argv[1:]) or "What is Agentic AI?"
    output = run_query(build_rag_graph(), question)
    output["retrieved_context_chunks"] = [t[:150] + "..." for t in output["retrieved_context_chunks"]]
    print(json.dumps(output, indent=2, ensure_ascii=False))
