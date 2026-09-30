"""Opt-in live check that the grounding grader can actually disagree with an answer.

The old grader asked gpt-4o-mini for a single 0-1 float and got 1.0 on every in-scope query, which
made the 0.6-weighted grounding term a constant. These two tests are the regression guard: a
faithful answer must pass, and the same answer with one fabricated sentence must fail.

They cost a couple of OpenAI calls, so they are skipped unless you ask for them:

    RUN_LIVE_TESTS=1 pytest tests/test_grounding_live.py -q       # bash
    $env:RUN_LIVE_TESTS=1; pytest tests/test_grounding_live.py -q  # PowerShell

Only OPENAI_API_KEY is needed - these never touch Pinecone.
"""
import os

import pytest

from src import config
from src.graph import GROUNDING_PROMPT, GroundingGrade, format_chunks, grounding_from_claims

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not config.OPENAI_API_KEY,
    reason="live LLM test: set RUN_LIVE_TESTS=1 and OPENAI_API_KEY to run",
)

CONTEXT = [{
    "text": ("Agentic AI creates impact by learning continuously, focusing on goals, and acting "
             "independently to anticipate needs. Unlike a generative chatbot, which responds to a "
             "single prompt, an agentic system decomposes a goal into steps and pursues them."),
    "page": 11,
    "similarity": 0.71,
    "relevant": True,
}]

FAITHFUL = ("Agentic AI learns continuously, focuses on goals and acts independently to anticipate "
            "needs (p. 11). Unlike a generative chatbot answering one prompt, it breaks a goal into "
            "steps and pursues them (p. 11).")

# Same first sentence, then two claims the context never makes.
FABRICATED = FAITHFUL + (" The eBook reports that agentic deployments cut operating costs by 43% "
                         "within six months, and names healthcare as the fastest-adopting sector "
                         "(p. 11).")


def grade(answer: str) -> float:
    from langchain_openai import ChatOpenAI

    grader = ChatOpenAI(model=config.LLM_MODEL, temperature=0).with_structured_output(GroundingGrade)
    result = grader.invoke(GROUNDING_PROMPT.format(context=format_chunks(CONTEXT), answer=answer))
    return grounding_from_claims(result.claims)


def test_a_faithful_answer_clears_the_grounding_threshold():
    assert grade(FAITHFUL) >= config.GROUNDING_THRESHOLD


def test_fabricated_statistics_drop_the_answer_below_the_threshold():
    score = grade(FABRICATED)
    assert score < config.GROUNDING_THRESHOLD, (
        f"grader scored a fabricated 43%%-cost-saving claim at {score:.2f}; it is saturating again"
    )
