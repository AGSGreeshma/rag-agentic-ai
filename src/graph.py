"""
LangGraph RAG workflow:
retrieve -> grade_documents -> generate -> check_grounding -> finalize / refuse
"""
from typing import TypedDict

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

ANSWER_PROMPT = """You answer questions using ONLY the context below, taken from an eBook about Agentic AI.

Rules:
- Use only information in the context. Never use outside knowledge.
- If the context does not contain the answer, reply exactly: "{refusal}"
- - Cite pages inline, like (p. 12), using ONLY the page numbers shown in the
  [Passage N | page X] headers. Ignore any page numbers that appear inside the text itself.
- Be concise: a short paragraph or a few bullet points.

Context:
{context}

Question: {question}
Answer:"""

GROUNDING_PROMPT = """You are checking an AI-generated answer for hallucinations.

Context:
{context}

Answer:
{answer}

Score from 0.0 to 1.0 how well the answer is supported by the context:
1.0 = every claim is directly supported by the context
0.5 = some claims are supported, others are not
0.0 = the answer is not supported by the context
Rephrasing and page citations are fine as long as the meaning matches the context."""


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
    chunks: list[Chunk]
    answer: str
    grounding_score: float
    confidence_score: float
    attempts: int


class RelevanceGrade(BaseModel):
    relevant_ids: list[int] = Field(description="Numbers of the passages that help answer the question")


class GroundingGrade(BaseModel):
    score: float = Field(description="Groundedness from 0.0 to 1.0")
    reason: str = Field(description="One sentence explaining the score")


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_rag_graph():
    config.validate()
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    relevance_grader = llm.with_structured_output(RelevanceGrade)
    grounding_grader = llm.with_structured_output(GroundingGrade)

    def format_chunks(chunks: list[Chunk]) -> str:
        return "\n\n".join(
            f"[Passage {i} | page {c['page']}]\n{c['text']}" for i, c in enumerate(chunks, start=1)
        )

    def relevant_chunks(state: AgentState) -> list[Chunk]:
        return [c for c in state["chunks"] if c["relevant"]]

    # --- Nodes ---
    def retrieve(state: AgentState):
        results = vector_store.similarity_search_with_score(state["question"], k=config.TOP_K)
        chunks = [
            {
                "text": doc.page_content,
                "page": int(doc.metadata.get("page", 0)),
                "similarity": round(float(score), 4),
                "relevant": False,
            }
            for doc, score in results
        ]
        return {"chunks": chunks}

    def grade_documents(state: AgentState):
        candidates = [c for c in state["chunks"] if c["similarity"] >= config.MIN_SIMILARITY]
        if not candidates:
            return {"chunks": state["chunks"]}
        grade = relevance_grader.invoke(
            RELEVANCE_PROMPT.format(question=state["question"], chunks=format_chunks(candidates))
        )
        keep = {candidates[i - 1]["text"] for i in grade.relevant_ids if 1 <= i <= len(candidates)}
        return {"chunks": [{**c, "relevant": c["text"] in keep} for c in state["chunks"]]}

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
        return {"answer": answer, "attempts": state["attempts"] + 1}

    def check_grounding(state: AgentState):
        if is_refusal(state["answer"]):
            return {"grounding_score": 0.0}
        grade = grounding_grader.invoke(
            GROUNDING_PROMPT.format(context=format_chunks(relevant_chunks(state)), answer=state["answer"])
        )
        return {"grounding_score": max(0.0, min(1.0, grade.score))}

    def finalize(state: AgentState):
        relevant = relevant_chunks(state)
        retrieval = sum(c["similarity"] for c in relevant) / len(relevant)
        confidence = (config.RETRIEVAL_WEIGHT * retrieval
                      + (1 - config.RETRIEVAL_WEIGHT) * state["grounding_score"])
        return {"confidence_score": round(confidence, 2)}

    def refuse(state: AgentState):
        return {"answer": REFUSAL, "confidence_score": 0.0}

    # --- Routing ---
    def route_after_grading(state: AgentState) -> str:
        return "generate" if relevant_chunks(state) else "refuse"

    def route_after_grounding(state: AgentState) -> str:
        if is_refusal(state["answer"]):
            return "refuse"
        if state["grounding_score"] >= config.GROUNDING_THRESHOLD:
            return "finalize"
        if state["attempts"] < config.MAX_GENERATION_ATTEMPTS:
            return "generate"
        return "refuse"

    # --- Wiring ---
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("check_grounding", check_grounding)
    workflow.add_node("finalize", finalize)
    workflow.add_node("refuse", refuse)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", route_after_grading,
                                   {"generate": "generate", "refuse": "refuse"})
    workflow.add_edge("generate", "check_grounding")
    workflow.add_conditional_edges("check_grounding", route_after_grounding,
                                   {"finalize": "finalize", "generate": "generate", "refuse": "refuse"})
    workflow.add_edge("finalize", END)
    workflow.add_edge("refuse", END)
    return workflow.compile()


# ---------------------------------------------------------------------------
# Public helper: run a query and shape the response
# ---------------------------------------------------------------------------
def run_query(graph, question: str) -> dict:
    """Invoke the graph and shape the output into the required response format."""
    result = graph.invoke({
        "question": question, "chunks": [], "answer": "",
        "grounding_score": 0.0, "confidence_score": 0.0, "attempts": 0,
    })
    return {
        "query": question,
        "final_answer": result["answer"],
        "retrieved_context_chunks": [c["text"] for c in result["chunks"]],
        "confidence_score": result["confidence_score"],
        "sources": [
            {"page": c["page"], "similarity": c["similarity"], "used_in_answer": c["relevant"]}
            for c in result["chunks"]
        ],
    }


if __name__ == "__main__":
    import json
    import sys

    question = " ".join(sys.argv[1:]) or "What is Agentic AI?"
    output = run_query(build_rag_graph(), question)
    output["retrieved_context_chunks"] = [t[:150] + "..." for t in output["retrieved_context_chunks"]]
    print(json.dumps(output, indent=2, ensure_ascii=False))