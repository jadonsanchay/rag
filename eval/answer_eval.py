"""Grade generated answers, not just retrieval.

Retrieval metrics say whether the right file was found. They say nothing about
whether the answer is grounded in it, whether its citations are real, or whether
the system admits ignorance when the context does not contain the answer. Those
are graded here.

Usage:
    uv run python eval/answer_eval.py --label step6
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config  # noqa: E402
from pipeline.generator import AnswerGenerator  # noqa: E402
from run_eval import load_golden  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from query import build_retriever  # noqa: E402


def grade(question: Dict[str, Any], answer, repo_root: Path) -> Dict[str, Any]:
    expected = set(question.get("expected_files") or [])
    answerable = bool(expected)
    verification = answer.verification

    cited_paths = answer.cited_paths

    def satisfies(path: str) -> bool:
        # A package card cites a directory; credit it when an expected file is inside.
        return path in expected or (
            path.endswith("/") and any(f.startswith(path) for f in expected)
        )

    grounded = any(satisfies(path) for path in cited_paths)
    retrieved_paths = [d["metadata"].get("path") for d in answer.sources]
    context_had_answer = any(satisfies(p) for p in retrieved_paths if p)

    # A refusal is only the generator's fault when the right file was actually in
    # its context. Refusing when retrieval missed is correct behaviour, and
    # lumping the two together blames the wrong component.
    refusal_warranted = answer.refused and not context_had_answer
    refusal_unwarranted = answer.refused and context_had_answer

    return {
        "id": question["id"],
        "type": question["type"],
        "answerable": answerable,
        "refused": answer.refused,
        "has_citations": bool(answer.cited_indices),
        "cited_indices": answer.cited_indices,
        "cited_paths": cited_paths,
        "grounded_in_expected": grounded if answerable else None,
        "context_had_answer": context_had_answer if answerable else None,
        "refusal_warranted": refusal_warranted if answerable else None,
        "refusal_unwarranted": refusal_unwarranted if answerable else None,
        "citations_checked": len(verification.checks) if verification else 0,
        "citations_valid": verification.valid_citations if verification else 0,
        "fabricated_indices": verification.fabricated_indices if verification else [],
        "retrieved_paths": [d["metadata"].get("path") for d in answer.sources],
        "answer": answer.text,
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def rate(subset, predicate) -> float:
        return round(sum(1 for r in subset if predicate(r)) / len(subset), 3) if subset else 0.0

    checked = sum(r["citations_checked"] for r in rows)
    valid = sum(r["citations_valid"] for r in rows)
    fabricated = sum(len(r["fabricated_indices"]) for r in rows)

    return {
        "answerable": {
            "n": len(answerable),
            "refusal_rate": rate(answerable, lambda r: r["refused"]),
            # The metric that actually indicts the generator: it had the right
            # file in context and still refused.
            "unwarranted_refusal_rate": rate(answerable, lambda r: r["refusal_unwarranted"]),
            # Refused because retrieval missed — correct behaviour, upstream fault.
            "warranted_refusal_rate": rate(answerable, lambda r: r["refusal_warranted"]),
            "citation_rate": rate(answerable, lambda r: r["has_citations"]),
            "grounded_rate": rate(answerable, lambda r: r["grounded_in_expected"]),
            "context_had_answer_rate": rate(answerable, lambda r: r["context_had_answer"]),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "refusal_rate": rate(unanswerable, lambda r: r["refused"]),
            "hallucination_rate": rate(unanswerable, lambda r: not r["refused"]),
        },
        "citations": {
            "checked": checked,
            "valid": valid,
            "validity_rate": round(valid / checked, 3) if checked else 0.0,
            "fabricated": fabricated,
        },
        "by_type": {
            qtype: {
                "n": len(subset),
                "false_refusal_rate": rate(subset, lambda r: r["refused"])
                if subset[0]["answerable"]
                else None,
                "grounded_rate": rate(subset, lambda r: r["grounded_in_expected"])
                if subset[0]["answerable"]
                else None,
            }
            for qtype, subset in sorted(_by_type(rows).items())
        },
    }


def _by_type(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row)
    return grouped


def print_report(summary: Dict[str, Any], label: str) -> None:
    print(f"\n=== {label} ===")
    ans, unans, cites = summary["answerable"], summary["unanswerable"], summary["citations"]

    print(f"\nAnswerable questions (n={ans['n']}):")
    print(f"  refusal rate            : {ans['refusal_rate']:.3f}")
    print(f"    unwarranted (gen bug) : {ans['unwarranted_refusal_rate']:.3f}  <- had the file, refused anyway")
    print(f"    warranted (retrieval) : {ans['warranted_refusal_rate']:.3f}  <- context lacked the answer")
    print(f"  context had answer      : {ans['context_had_answer_rate']:.3f}")
    print(f"  citation rate           : {ans['citation_rate']:.3f}")
    print(f"  grounded rate           : {ans['grounded_rate']:.3f}  (cites an expected file)")

    print(f"\nUnanswerable questions (n={unans['n']}):")
    print(f"  refusal rate       : {unans['refusal_rate']:.3f}  (higher is better)")
    print(f"  hallucination rate : {unans['hallucination_rate']:.3f}")

    print(f"\nCitations: {cites['valid']}/{cites['checked']} verified "
          f"({cites['validity_rate']:.3f}), {cites['fabricated']} fabricated")

    print(f"\n{'type':<16}{'n':>4}{'false refusal':>15}{'grounded':>10}")
    for qtype, stats in summary["by_type"].items():
        fr = "-" if stats["false_refusal_rate"] is None else f"{stats['false_refusal_rate']:.3f}"
        gr = "-" if stats["grounded_rate"] is None else f"{stats['grounded_rate']:.3f}"
        print(f"{qtype:<16}{stats['n']:>4}{fr:>15}{gr:>10}")


def main():
    parser = argparse.ArgumentParser(description="Grade generated answers")
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo", default="fastapi")
    parser.add_argument("--variant", default="astcode-cards")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Grade only the first N")
    args = parser.parse_args()

    questions = load_golden(include_unanswerable=True)
    if args.limit:
        questions = questions[: args.limit]

    repo_root = config.REPOS_DIR / args.repo
    retriever = build_retriever(args.repo, args.variant)
    generator = AnswerGenerator()

    rows = []
    for index, question in enumerate(questions, start=1):
        results = retriever.retrieve(question["question"], top_k=args.top_k)
        answer = generator.generate(question["question"], results, repo_root=repo_root)
        rows.append(grade(question, answer, repo_root))
        print(f"  graded {index}/{len(questions)}", end="\r", flush=True)
    print(" " * 40, end="\r")

    summary = summarize(rows)
    print_report(summary, args.label)

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_RESULTS_DIR / f"answers-{args.label}.json"
    out_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "repo": args.repo,
                "variant": args.variant,
                "top_k": args.top_k,
                "generator_model": generator.model,
                "summary": summary,
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"\nSaved: {out_path.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
