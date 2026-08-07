"""Measure retrieval quality against the golden question set.

Usage:
    uv run python eval/run_eval.py --label baseline-text-minilm
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config  # noqa: E402
from pipeline.embeddings import EmbeddingManager  # noqa: E402
from pipeline.retriever import RAGRetriever  # noqa: E402
from pipeline.vector_store import VectorStore, collection_name_for_repo  # noqa: E402

GOLDEN_PATH = config.EVAL_DIR / "golden.jsonl"
RECALL_KS = (1, 5, 10)
RETRIEVE_K = 20


def load_golden(path: Path = GOLDEN_PATH) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_hit_rank(retrieved_paths: Sequence[str], expected_files: Sequence[str]) -> Optional[int]:
    """1-indexed rank of the first retrieved chunk from an expected file."""
    expected = set(expected_files)
    for index, path in enumerate(retrieved_paths, start=1):
        if path in expected:
            return index
    return None


def evaluate_question(
    retriever: RAGRetriever, question: Dict[str, Any], retrieve_k: int
) -> Dict[str, Any]:
    results = retriever.retrieve(question["question"], top_k=retrieve_k)
    retrieved_paths = [r["metadata"].get("path", "") for r in results]
    rank = first_hit_rank(retrieved_paths, question["expected_files"])

    return {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "expected_files": question["expected_files"],
        "first_hit_rank": rank,
        "hit_at": {k: bool(rank and rank <= k) for k in RECALL_KS},
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "distinct_files_retrieved": len(dict.fromkeys(retrieved_paths[:10])),
        "top_5_paths": retrieved_paths[:5],
    }


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {}
        n = len(subset)
        summary = {"n": n}
        for k in RECALL_KS:
            summary[f"recall@{k}"] = round(
                sum(row["hit_at"][k] for row in subset) / n, 3
            )
        summary["mrr"] = round(sum(row["reciprocal_rank"] for row in subset) / n, 3)
        summary["misses"] = [row["id"] for row in subset if row["first_hit_rank"] is None]
        return summary

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    return {
        "overall": summarize(rows),
        "by_type": {qtype: summarize(subset) for qtype, subset in sorted(by_type.items())},
    }


def print_report(summary: Dict[str, Any], label: str) -> None:
    overall = summary["overall"]
    header = f"{'segment':<16}{'n':>4}" + "".join(f"{f'r@{k}':>8}" for k in RECALL_KS) + f"{'MRR':>8}"
    print(f"\n=== {label} ===")
    print(header)
    print("-" * len(header))

    def row(name: str, stats: Dict[str, Any]) -> None:
        cells = "".join(f"{stats[f'recall@{k}']:>8.3f}" for k in RECALL_KS)
        print(f"{name:<16}{stats['n']:>4}{cells}{stats['mrr']:>8.3f}")

    row("overall", overall)
    for qtype, stats in summary["by_type"].items():
        row(qtype, stats)

    if overall["misses"]:
        print(f"\nComplete misses (not found in top {RETRIEVE_K}): {', '.join(overall['misses'])}")


def load_manifest(collection: str) -> Dict[str, Any]:
    path = config.DATA_DIR / "index_manifests" / f"{collection}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--label", required=True, help="Name for this run's results file")
    parser.add_argument("--repo", default="fastapi")
    parser.add_argument("--strategy", default="text", choices=["text", "ast", "ast-code"])
    parser.add_argument("--retrieve-k", type=int, default=RETRIEVE_K)
    parser.add_argument(
        "--provider",
        default=config.EMBEDDING_PROVIDER,
        choices=["sentence-transformers", "openai"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--variant", default=None, help="Collection suffix (defaults to --strategy)"
    )
    args = parser.parse_args()

    questions = load_golden()
    variant = args.variant or args.strategy
    collection = collection_name_for_repo(args.repo, variant)

    # The query must be embedded by the same model that built the index.
    embedder = EmbeddingManager(model_name=args.model, provider=args.provider)
    retriever = RAGRetriever(VectorStore(collection_name=collection), embedder)

    rows = [evaluate_question(retriever, q, args.retrieve_k) for q in questions]
    summary = aggregate(rows)
    print_report(summary, args.label)

    payload = {
        "label": args.label,
        "repo": args.repo,
        "strategy": args.strategy,
        "variant": variant,
        "collection": collection,
        "retrieve_k": args.retrieve_k,
        "questions": len(questions),
        "config": {
            "embedding_provider": args.provider,
            "embedding_model": embedder.model_name,
            "token_limit": embedder.token_limit,
        },
        "index_manifest": load_manifest(collection),
        "summary": summary,
        "results": rows,
    }

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_RESULTS_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
