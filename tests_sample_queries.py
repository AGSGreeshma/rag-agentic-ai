"""
Runs the sample validation queries through the RAG pipeline and checks grounding.

Usage:
    python tests_sample_queries.py          # call the LangGraph pipeline directly
    python tests_sample_queries.py --api    # call the running FastAPI server instead

Full results are saved to results/sample_query_results.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from src.graph import build_rag_graph, run_query

API_URL = "http://127.0.0.1:8000/chat"
RESULTS_PATH = Path(__file__).parent / "results" / "sample_query_results.json"
MIN_CONFIDENCE = 0.5

TEST_CASES = [
    # From the assignment brief
    {"query": "What is the core definition of Agentic AI as outlined in the eBook?", "in_scope": True},
    {"query": "What are the main architectural components required to build agentic systems?", "in_scope": True},
    {"query": "What real-world industry use cases for Agentic AI are discussed in the eBook?", "in_scope": True},
    {"query": "How does Agentic AI differ from traditional generative AI chatbots according to the text?", "in_scope": True},
    {"query": "What key challenges or limitations of Agentic AI are mentioned in the document?", "in_scope": True},
    {"query": "What is the capital of France?", "in_scope": False},
    # From the reference guide
    {"query": "What role does memory play in Agentic AI workflows?", "in_scope": True},
    {"query": "Who won the 2022 FIFA World Cup?", "in_scope": False},
]


def ask_api(query: str) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def check(case: dict, result: dict) -> list[str]:
    """Return a list of problems (empty list = pass)."""
    problems = []
    # The pipeline records the refusal itself; trust that over re-matching the answer text.
    refused = result["out_of_scope"]
    confidence = result["confidence_score"]

    if case["in_scope"]:
        if refused:
            problems.append("refused an in-scope question")
        if confidence < MIN_CONFIDENCE:
            problems.append(f"confidence {confidence:.2f} is below {MIN_CONFIDENCE}")
        cited = {int(p) for p in re.findall(r"p\.\s*(\d+)", result["final_answer"])}
        retrieved = {s["page"] for s in result["sources"]}
        if not refused and not cited:
            problems.append("answer has no page citations")
        elif not cited <= retrieved:
            problems.append(f"cited pages {sorted(cited - retrieved)} were not in the retrieved context")
    else:
        if not refused:
            problems.append("answered an out-of-scope question instead of refusing")
        if confidence != 0.0:
            problems.append(f"confidence should be 0.0 for a refusal, got {confidence:.2f}")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Run sample queries against the RAG pipeline.")
    parser.add_argument("--api", action="store_true", help="Call the running FastAPI server")
    args = parser.parse_args()

    graph = None if args.api else build_rag_graph()
    records, failures = [], 0

    for i, case in enumerate(TEST_CASES, start=1):
        start = time.perf_counter()
        result = ask_api(case["query"]) if args.api else run_query(graph, case["query"])
        elapsed = time.perf_counter() - start

        problems = check(case, result)
        status = "PASS" if not problems else "FAIL"
        failures += bool(problems)

        used_pages = [s["page"] for s in result["sources"] if s["used_in_answer"]]
        print(f"\n[{i}/{len(TEST_CASES)}] {status}  {case['query']}")
        print(f"  confidence={result['confidence_score']:.2f}  time={elapsed:.1f}s  pages used={used_pages}")
        print(f"  answer: {result['final_answer'][:200]}{'...' if len(result['final_answer']) > 200 else ''}")
        for p in problems:
            print(f"  ! {p}")

        records.append({
            **result,
            "expected_in_scope": case["in_scope"],
            "status": status,
            "problems": problems,
            "latency_seconds": round(elapsed, 2),
        })

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    passed = len(TEST_CASES) - failures
    print(f"\n{'=' * 60}\n{passed}/{len(TEST_CASES)} passed. Full results: {RESULTS_PATH}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()