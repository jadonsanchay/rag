"""Does query rewriting actually make follow-ups retrievable?

The claim behind step 12 is that a follow-up like "and where is that called from?"
cannot be retrieved as written, and becomes retrievable once condensed against the
history. That is testable: run each follow-up through retrieval twice — raw and
rewritten — and compare whether the expected file surfaces.

Usage:
    uv run python eval/rewrite_eval.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config  # noqa: E402
from pipeline.conversation import QueryRewriter  # noqa: E402
from query import build_retriever  # noqa: E402

TOP_K = 10

# Each case is a real two-turn exchange against the indexed fastapi corpus. The
# follow-up is written the way someone actually types it, and `expected_files` is
# what a correct retrieval must surface.
CASES: List[Dict[str, Any]] = [
    {
        "id": "r1",
        "history": [
            {"role": "user", "content": "Where is the APIRouter class defined?"},
            {"role": "assistant", "content": "APIRouter is defined in fastapi/routing.py, extending Starlette's routing.Router."},
        ],
        "follow_up": "and where is it included into an app?",
        "expected_files": ["fastapi/applications.py", "fastapi/routing.py"],
    },
    {
        "id": "r2",
        "history": [
            {"role": "user", "content": "How does FastAPI turn Python objects into JSON?"},
            {"role": "assistant", "content": "It uses jsonable_encoder, which converts Pydantic models and other objects into JSON-serializable data."},
        ],
        "follow_up": "which file is that in?",
        "expected_files": ["fastapi/encoders.py"],
    },
    {
        "id": "r3",
        "history": [
            {"role": "user", "content": "What handles validation errors?"},
            {"role": "assistant", "content": "RequestValidationError is raised and handled by request_validation_exception_handler."},
        ],
        "follow_up": "what status code does it return?",
        "expected_files": ["fastapi/exception_handlers.py", "fastapi/exceptions.py"],
    },
    {
        "id": "r4",
        "history": [
            {"role": "user", "content": "How are dependencies resolved for a request?"},
            {"role": "assistant", "content": "solve_dependencies walks the Dependant tree and resolves each sub-dependency."},
        ],
        "follow_up": "does it cache them?",
        "expected_files": ["fastapi/dependencies/utils.py"],
    },
    {
        "id": "r5",
        "history": [
            {"role": "user", "content": "How is the OpenAPI schema produced?"},
            {"role": "assistant", "content": "get_openapi builds the document from the app's routes."},
        ],
        "follow_up": "and how does the Swagger page get served?",
        "expected_files": ["fastapi/openapi/docs.py", "fastapi/applications.py"],
    },
    {
        "id": "r6",
        "history": [
            {"role": "user", "content": "What is BackgroundTasks for?"},
            {"role": "assistant", "content": "It queues work to run after the response has been sent."},
        ],
        "follow_up": "where is that class defined?",
        "expected_files": ["fastapi/background.py"],
    },
    {
        "id": "r7",
        "history": [
            {"role": "user", "content": "How does FastAPI support both Pydantic v1 and v2?"},
            {"role": "assistant", "content": "A compatibility layer branches on the installed version."},
        ],
        "follow_up": "which module holds it?",
        "expected_files": ["fastapi/_compat.py"],
    },
    {
        "id": "r8",
        "history": [
            {"role": "user", "content": "How does an endpoint run if it is a normal def?"},
            {"role": "assistant", "content": "FastAPI runs it in a threadpool so it does not block the event loop."},
        ],
        "follow_up": "what does it use to do that?",
        "expected_files": ["fastapi/routing.py", "fastapi/concurrency.py", "fastapi/dependencies/utils.py"],
    },
]


def hit_rank(retriever, question: str, expected: List[str]) -> Any:
    results = retriever.retrieve(question, top_k=TOP_K)
    expected_set = set(expected)
    for index, result in enumerate(results, start=1):
        path = result["metadata"].get("path", "")
        if path in expected_set or (
            path.endswith("/") and any(f.startswith(path) for f in expected_set)
        ):
            return index
    return None


def main() -> None:
    retriever = build_retriever("fastapi", "astcode-cards")
    rewriter = QueryRewriter()

    rows = []
    print(f"{'id':<4}{'raw':>6}{'rewritten':>11}   follow-up -> rewritten query")
    print("-" * 100)

    for case in CASES:
        raw_rank = hit_rank(retriever, case["follow_up"], case["expected_files"])
        result = rewriter.rewrite(case["follow_up"], case["history"])
        new_rank = hit_rank(retriever, result.query, case["expected_files"])

        rows.append(
            {
                "id": case["id"],
                "follow_up": case["follow_up"],
                "rewritten_query": result.query,
                "was_rewritten": result.rewritten,
                "reason": result.reason,
                "raw_rank": raw_rank,
                "rewritten_rank": new_rank,
            }
        )
        print(
            f"{case['id']:<4}{str(raw_rank or 'miss'):>6}{str(new_rank or 'miss'):>11}"
            f"   {case['follow_up']} -> {result.query}"
        )

    def recall(key: str, k: int) -> float:
        hits = sum(1 for r in rows if r[key] and r[key] <= k)
        return round(hits / len(rows), 3)

    def mrr(key: str) -> float:
        return round(sum(1 / r[key] if r[key] else 0 for r in rows) / len(rows), 3)

    summary = {
        "n": len(rows),
        "raw": {"recall@5": recall("raw_rank", 5), "recall@10": recall("raw_rank", 10), "mrr": mrr("raw_rank")},
        "rewritten": {
            "recall@5": recall("rewritten_rank", 5),
            "recall@10": recall("rewritten_rank", 10),
            "mrr": mrr("rewritten_rank"),
        },
        "rewrite_applied": sum(1 for r in rows if r["was_rewritten"]),
    }

    print(f"\n{'':12}{'r@5':>8}{'r@10':>8}{'MRR':>8}")
    for label in ("raw", "rewritten"):
        s = summary[label]
        print(f"{label:<12}{s['recall@5']:>8.3f}{s['recall@10']:>8.3f}{s['mrr']:>8.3f}")
    print(f"\nrewrite applied to {summary['rewrite_applied']}/{len(rows)} follow-ups")

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EVAL_RESULTS_DIR / "rewrite-eval.json"
    out.write_text(json.dumps({"summary": summary, "results": rows}, indent=2))
    print(f"Saved: {out.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
