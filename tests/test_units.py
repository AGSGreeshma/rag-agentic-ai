"""Unit tests for the pipeline's decision logic and arithmetic.

These never touch OpenAI or Pinecone: every function under test takes a plain state dict, so the
whole file runs offline, instantly, for free, and deterministically. `tests_sample_queries.py`
remains the end-to-end evidence; this file covers the branches a live run never reaches.
"""
import pytest

from src import config
from src.graph import (
    REFUSAL,
    ClaimVerdict,
    compute_confidence,
    format_chunks,
    format_result,
    grounding_from_claims,
    initial_state,
    is_refusal,
    relevant_chunks,
    retrieval_score,
    route_after_grading,
    route_after_grounding,
)
from src.ingestion import clean_text


# --- Builders ---------------------------------------------------------------
def chunk(text="a passage", page=1, similarity=0.70, relevant=True):
    return {"text": text, "page": page, "similarity": similarity, "relevant": relevant}


def state(**overrides):
    """A starting state with the given fields overridden."""
    s = initial_state(overrides.pop("question", "What is Agentic AI?"))
    s.update(overrides)
    return s


def claims(*verdicts):
    return [ClaimVerdict(claim=f"claim {i}", verdict=v) for i, v in enumerate(verdicts)]


# --- clean_text -------------------------------------------------------------
def test_clean_text_collapses_runs_of_spaces_and_tabs():
    assert clean_text("Agentic    AI\tsystems") == "Agentic AI systems"


def test_clean_text_collapses_large_blank_gaps_but_keeps_paragraphs():
    assert clean_text("first\n\n\n\n\nsecond") == "first\n\nsecond"
    assert clean_text("first\n\nsecond") == "first\n\nsecond"


def test_clean_text_strips_surrounding_whitespace():
    assert clean_text("\n  text  \n") == "text"


# --- is_refusal -------------------------------------------------------------
def test_is_refusal_matches_the_exact_message_and_ignores_case_and_period():
    assert is_refusal(REFUSAL)
    assert is_refusal(REFUSAL.rstrip("."))
    assert is_refusal(REFUSAL.upper())


def test_is_refusal_does_not_fire_on_a_real_answer():
    assert not is_refusal("Agentic AI learns continuously and acts independently (p. 11).")


# --- grounding_from_claims --------------------------------------------------
def test_grounding_is_one_when_every_claim_is_supported():
    assert grounding_from_claims(claims("supported", "supported", "supported")) == 1.0


def test_grounding_is_zero_when_no_claim_is_supported():
    assert grounding_from_claims(claims("unsupported", "unsupported")) == 0.0


def test_partial_claims_count_as_half():
    assert grounding_from_claims(claims("partial", "partial")) == 0.5
    assert grounding_from_claims(claims("supported", "unsupported")) == 0.5


def test_one_unsupported_claim_in_four_drops_below_the_grounding_threshold():
    """The point of claim-level scoring: a mostly-right answer must still be able to fail."""
    score = grounding_from_claims(claims("supported", "supported", "unsupported", "unsupported"))
    assert score < config.GROUNDING_THRESHOLD


def test_no_claims_extracted_fails_closed():
    """An empty verdict list means the grader told us nothing, so it must not read as grounded."""
    assert grounding_from_claims([]) == 0.0


# --- route_after_grading ----------------------------------------------------
def test_enough_relevant_chunks_go_straight_to_generate():
    s = state(chunks=[chunk("a"), chunk("b")])
    assert len(relevant_chunks(s)) >= config.MIN_RELEVANT_CHUNKS
    assert route_after_grading(s) == "generate"


def test_everything_below_the_similarity_floor_refuses_without_a_rewrite():
    """Clearly off-topic questions: the README's ~1s refusal path, with no LLM call."""
    s = state(chunks=[chunk("a", similarity=0.08, relevant=False),
                      chunk("b", similarity=0.05, relevant=False)])
    assert route_after_grading(s) == "refuse"


def test_too_few_relevant_chunks_on_the_first_pass_triggers_a_rewrite():
    s = state(chunks=[chunk("a"), chunk("b", relevant=False)])
    assert route_after_grading(s) == "rewrite_query"


def test_no_relevant_chunks_but_on_topic_still_tries_a_rewrite_first():
    s = state(chunks=[chunk("a", relevant=False), chunk("b", relevant=False)])
    assert route_after_grading(s) == "rewrite_query"


def test_one_relevant_chunk_is_answered_once_the_rewrite_budget_is_spent():
    """Better a partial answer from one good chunk than a refusal - the README's design call."""
    s = state(chunks=[chunk("a"), chunk("b", relevant=False)],
              rewrites=config.MAX_QUERY_REWRITES)
    assert route_after_grading(s) == "generate"


def test_nothing_relevant_after_the_rewrite_refuses():
    s = state(chunks=[chunk("a", relevant=False), chunk("b", relevant=False)],
              rewrites=config.MAX_QUERY_REWRITES)
    assert route_after_grading(s) == "refuse"


def test_generate_is_never_entered_with_an_empty_context():
    """Guards the division in compute_confidence: `generate` must imply at least one chunk."""
    for n_relevant in range(3):
        for rewrites in range(config.MAX_QUERY_REWRITES + 2):
            for similarity in (0.05, 0.70):
                chunks = [chunk(f"c{i}", similarity=similarity, relevant=i < n_relevant)
                          for i in range(3)]
                s = state(chunks=chunks, rewrites=rewrites)
                if route_after_grading(s) == "generate":
                    assert relevant_chunks(s), (n_relevant, rewrites, similarity)


# --- route_after_grounding --------------------------------------------------
def test_a_refusal_routes_to_refuse_regardless_of_score():
    s = state(refused=True, grounding_score=1.0, chunks=[chunk()])
    assert route_after_grounding(s) == "refuse"


def test_a_grounded_answer_is_finalized():
    s = state(grounding_score=config.GROUNDING_THRESHOLD, attempts=1, chunks=[chunk()])
    assert route_after_grounding(s) == "finalize"


def test_an_ungrounded_first_answer_is_retried():
    s = state(grounding_score=0.4, attempts=1, chunks=[chunk()])
    assert route_after_grounding(s) == "generate"


def test_an_ungrounded_answer_refuses_once_retries_run_out():
    s = state(grounding_score=0.4, attempts=config.MAX_GENERATION_ATTEMPTS, chunks=[chunk()])
    assert route_after_grounding(s) == "refuse"


# --- retrieval_score / compute_confidence -----------------------------------
def test_retrieval_score_is_the_mean_similarity_of_the_used_chunks_only():
    s = state(chunks=[chunk("a", similarity=0.80), chunk("b", similarity=0.60),
                      chunk("c", similarity=0.10, relevant=False)])
    assert retrieval_score(s) == 0.70


def test_retrieval_score_is_none_with_no_used_chunks():
    assert retrieval_score(state(chunks=[chunk(relevant=False)])) is None


def test_confidence_applies_the_configured_weights():
    s = state(chunks=[chunk(similarity=0.50)], grounding_score=1.0)
    expected = round(config.RETRIEVAL_WEIGHT * 0.50 + (1 - config.RETRIEVAL_WEIGHT) * 1.0, 2)
    assert compute_confidence(s) == expected


def test_confidence_is_zero_when_nothing_was_used():
    s = state(chunks=[chunk(relevant=False)], grounding_score=1.0)
    assert compute_confidence(s) == 0.0


def test_grounding_moves_confidence():
    """Regression for the saturated-grader bug: confidence must not be a function of retrieval alone."""
    grounded = state(chunks=[chunk(similarity=0.65)], grounding_score=1.0)
    ungrounded = state(chunks=[chunk(similarity=0.65)], grounding_score=0.5)
    assert compute_confidence(grounded) > compute_confidence(ungrounded)


# --- format_result ----------------------------------------------------------
ASSIGNMENT_KEYS = ["query", "final_answer", "retrieved_context_chunks", "confidence_score"]


def test_format_result_keeps_the_required_assignment_keys_first():
    s = state(answer="An answer (p. 7).", chunks=[chunk()], grounding_score=1.0,
              confidence_score=0.88)
    assert list(format_result("q", s))[:4] == ASSIGNMENT_KEYS


def test_format_result_reports_every_retrieved_chunk_and_flags_which_were_used():
    s = state(answer="An answer (p. 7).", grounding_score=1.0, confidence_score=0.88,
              chunks=[chunk("used", page=7), chunk("dropped", page=9, relevant=False)])
    out = format_result("q", s)
    assert out["retrieved_context_chunks"] == ["used", "dropped"]
    assert [src["used_in_answer"] for src in out["sources"]] == [True, False]
    assert [src["page"] for src in out["sources"]] == [7, 9]


def test_format_result_blanks_the_scores_on_a_refusal():
    s = state(answer=REFUSAL, refused=True, chunks=[chunk(relevant=False)])
    out = format_result("q", s)
    assert out["out_of_scope"] is True
    assert out["confidence_score"] == 0.0
    assert out["grounding_score"] is None
    assert out["retrieval_score"] is None


def test_format_result_trusts_the_refused_flag_over_the_answer_text():
    """A paraphrased refusal is recorded once, in `generate`, and honoured everywhere after."""
    s = state(answer="I can't answer that from this document.", refused=True,
              chunks=[chunk(relevant=False)])
    assert format_result("q", s)["out_of_scope"] is True


def test_format_result_exposes_the_rewritten_queries_when_the_loop_ran():
    s = state(answer="An answer (p. 36).", chunks=[chunk(page=36)], grounding_score=1.0,
              confidence_score=0.85, rewrites=1,
              search_queries=["challenges, risks, barriers", "mitigation strategies"])
    assert format_result("q", s)["rewritten_queries"] == [
        "challenges, risks, barriers", "mitigation strategies"]


# --- format_chunks ----------------------------------------------------------
def test_format_chunks_numbers_passages_from_one_and_labels_the_page():
    out = format_chunks([chunk("first", page=7), chunk("second", page=11)])
    assert "[Passage 1 | page 7]" in out
    assert "[Passage 2 | page 11]" in out


def test_format_chunks_is_empty_for_no_chunks():
    assert format_chunks([]) == ""


@pytest.mark.parametrize("weight_name", ["RETRIEVAL_WEIGHT", "MIN_SIMILARITY"])
def test_tuning_constants_are_fractions(weight_name):
    assert 0.0 <= getattr(config, weight_name) <= 1.0
