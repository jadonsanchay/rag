"""Sweep fusion weights and RRF k, printing a comparison table.

Prints only; use run_eval.py --label to persist the configuration you pick.

Usage:
    uv run python eval/sweep.py --variant astcode-openai --provider openai
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config  # noqa: E402
from pipeline.embeddings import EmbeddingManager  # noqa: E402
from pipeline.lexical_index import LexicalIndex  # noqa: E402
from pipeline.retriever import HybridRetriever  # noqa: E402
from pipeline.vector_store import VectorStore, collection_name_for_repo  # noqa: E402
from run_eval import RECALL_KS, aggregate, evaluate_question, load_golden  # noqa: E402

# (semantic_weight, lexical_weight) — includes the pure modes as reference rows.
WEIGHT_GRID = [
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (1.0, 3.0),
    (1.0, 5.0),
    (0.5, 1.0),
    (0.25, 1.0),
]
RRF_KS = [10, 60]


def main():
    parser = argparse.ArgumentParser(description="Sweep hybrid fusion parameters")
    parser.add_argument("--repo", default="fastapi")
    parser.add_argument("--variant", default="astcode-openai")
    parser.add_argument(
        "--provider", default=config.EMBEDDING_PROVIDER,
        choices=["sentence-transformers", "openai"],
    )
    parser.add_argument("--retrieve-k", type=int, default=20)
    args = parser.parse_args()

    questions = load_golden()
    collection = collection_name_for_repo(args.repo, args.variant)
    embedder = EmbeddingManager(provider=args.provider)
    store = VectorStore(collection_name=collection)
    lexical = LexicalIndex(collection)

    segments = ["pinpoint", "behavioral", "architectural", "paraphrase"]
    header = (
        f"{'sem':>5}{'lex':>5}{'k':>5}"
        + "".join(f"{f'r@{k}':>8}" for k in RECALL_KS)
        + f"{'MRR':>8}"
        + "".join(f"{s[:6]:>8}" for s in segments)
    )
    print(header)
    print("-" * len(header))

    best = None
    for rrf_k in RRF_KS:
        for semantic_weight, lexical_weight in WEIGHT_GRID:
            if semantic_weight == 0.0:
                mode = "lexical"
            elif lexical_weight == 0.0:
                mode = "semantic"
            else:
                mode = "hybrid"
            # Pure modes ignore k; only report them once.
            if mode != "hybrid" and rrf_k != RRF_KS[0]:
                continue

            retriever = HybridRetriever(
                store,
                embedder,
                lexical_index=lexical if mode != "semantic" else None,
                mode=mode,
                semantic_weight=semantic_weight,
                lexical_weight=lexical_weight,
                rrf_k=rrf_k,
                candidate_k=max(40, args.retrieve_k),
            )
            rows = [evaluate_question(retriever, q, args.retrieve_k) for q in questions]
            summary = aggregate(rows)
            overall = summary["overall"]
            by_type = summary["by_type"]

            cells = "".join(f"{overall[f'recall@{k}']:>8.3f}" for k in RECALL_KS)
            per_segment = "".join(
                f"{by_type.get(s, {}).get('mrr', 0.0):>8.3f}" for s in segments
            )
            label_k = rrf_k if mode == "hybrid" else "-"
            print(
                f"{semantic_weight:>5}{lexical_weight:>5}{label_k:>5}{cells}"
                f"{overall['mrr']:>8.3f}{per_segment}"
            )

            if best is None or overall["mrr"] > best[0]:
                best = (overall["mrr"], semantic_weight, lexical_weight, rrf_k, mode)

    print(f"\nBest overall MRR: {best[0]:.3f} "
          f"(mode={best[4]}, semantic={best[1]}, lexical={best[2]}, k={best[3]})")


if __name__ == "__main__":
    main()
