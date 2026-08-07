"""Diff two eval runs so every change is justified by a number.

Usage:
    uv run python eval/compare.py baseline-text-minilm ast-openai
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config  # noqa: E402

RECALL_KS = (1, 5, 10)


def load_run(label: str) -> Dict[str, Any]:
    path = config.EVAL_RESULTS_DIR / f"{label}.json"
    if not path.exists():
        raise SystemExit(f"No results for '{label}' at {path}")
    return json.loads(path.read_text())


def fmt_delta(before: float, after: float) -> str:
    delta = after - before
    arrow = "+" if delta > 0 else ""
    return f"{before:.3f} -> {after:.3f} ({arrow}{delta:.3f})"


def print_config_diff(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    print("\nConfig:")
    for key in ("strategy", "collection"):
        if before.get(key) != after.get(key):
            print(f"  {key}: {before.get(key)} -> {after.get(key)}")
    b_cfg, a_cfg = before.get("config", {}), after.get("config", {})
    for key in sorted(set(b_cfg) | set(a_cfg)):
        if b_cfg.get(key) != a_cfg.get(key):
            print(f"  {key}: {b_cfg.get(key)} -> {a_cfg.get(key)}")

    b_man = before.get("index_manifest", {})
    a_man = after.get("index_manifest", {})
    for key in ("chunks", "files_indexed"):
        if b_man.get(key) != a_man.get(key):
            print(f"  {key}: {b_man.get(key)} -> {a_man.get(key)}")
    b_dup = (b_man.get("duplicates") or {}).get("pct_duplicate")
    a_dup = (a_man.get("duplicates") or {}).get("pct_duplicate")
    if b_dup != a_dup:
        print(f"  duplicate chunks %: {b_dup} -> {a_dup}")
    b_over = (b_man.get("token_budget") or {}).get("pct_over_limit")
    a_over = (a_man.get("token_budget") or {}).get("pct_over_limit")
    if b_over != a_over:
        print(f"  chunks over token limit %: {b_over} -> {a_over}")


def print_metric_table(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    segments = ["overall"] + [f"by_type.{t}" for t in sorted(after["summary"]["by_type"])]

    def get(run: Dict[str, Any], segment: str) -> Dict[str, Any]:
        if segment == "overall":
            return run["summary"]["overall"]
        return run["summary"]["by_type"][segment.split(".", 1)[1]]

    print(f"\n{'segment':<16}{'metric':<8}{'change':>28}")
    print("-" * 52)
    for segment in segments:
        b, a = get(before, segment), get(after, segment)
        name = segment.replace("by_type.", "")
        for k in RECALL_KS:
            key = f"recall@{k}"
            print(f"{name:<16}{f'r@{k}':<8}{fmt_delta(b[key], a[key]):>28}")
        print(f"{name:<16}{'MRR':<8}{fmt_delta(b['mrr'], a['mrr']):>28}")


def print_per_question(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    b_rows = {row["id"]: row for row in before["results"]}
    improved, regressed = [], []

    for row in after["results"]:
        b_row = b_rows.get(row["id"])
        if not b_row:
            continue
        b_rank = b_row["first_hit_rank"]
        a_rank = row["first_hit_rank"]
        if b_rank == a_rank:
            continue
        # None means "never found"; treat as worse than any real rank.
        b_score = b_rank if b_rank else 999
        a_score = a_rank if a_rank else 999
        entry = (row["id"], row["type"], b_rank, a_rank)
        (improved if a_score < b_score else regressed).append(entry)

    def show(title: str, entries: list) -> None:
        if not entries:
            return
        print(f"\n{title}:")
        for qid, qtype, b_rank, a_rank in entries:
            print(f"  {qid} ({qtype}): rank {b_rank or 'miss'} -> {a_rank or 'miss'}")

    show("Improved", improved)
    show("Regressed", regressed)
    if not improved and not regressed:
        print("\nNo per-question rank changes.")


def main():
    parser = argparse.ArgumentParser(description="Compare two eval runs")
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before, after = load_run(args.before), load_run(args.after)
    print(f"=== {args.before}  ->  {args.after} ===")
    print_config_diff(before, after)
    print_metric_table(before, after)
    print_per_question(before, after)


if __name__ == "__main__":
    main()
