"""Run the faithfulness judge, and validate the judge itself.

    uv run python eval/run_judge.py --validate        # can it catch known-bad claims?
    uv run python eval/run_judge.py --label step6     # grade the real answers

The --validate pass runs first for a reason: a judge that cannot detect an
injected falsehood produces numbers that mean nothing, and "the answers scored
0.95 faithful" would be indistinguishable from "the judge rubber-stamps
everything".
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config  # noqa: E402
from judge import FaithfulnessJudge, aggregate  # noqa: E402
from query import build_retriever  # noqa: E402

# Claims that no source in a FastAPI corpus can support. Each names something
# absent from the repository entirely, so "unsupported" is not a judgement call.
# The keyword is how detection is confirmed: comparing unsupported *counts* before
# and after is confounded, because the judge's claim decomposition is unstable
# (observed 13 -> 15 claims on the same answer). Looking for the injected claim
# specifically is immune to that.
PERTURBATIONS = [
    ("This behaviour is controlled by the FASTAPI_STRICT_MODE environment variable.",
     "strict_mode"),
    ("Results are memoised in a Redis cache with a sixty second time-to-live.",
     "redis"),
    ("The implementation delegates to the billing module to record usage.",
     "billing"),
]


def load_golden_by_id() -> Dict[str, Dict[str, Any]]:
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in (config.EVAL_DIR / "golden.jsonl").read_text().splitlines()
        if line.strip()
    }


def load_answers(label: str) -> Dict[str, Any]:
    path = config.EVAL_RESULTS_DIR / f"answers-{label}.json"
    if not path.exists():
        raise SystemExit(f"No answer run at {path}. Run answer_eval.py first.")
    return json.loads(path.read_text())


def gradeable(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only answerable, non-refused answers have claims worth grading."""
    return [r for r in rows if r["answerable"] and not r["refused"] and r["answer"].strip()]


def stratified_sample(rows: Sequence[Dict[str, Any]], sample: int) -> List[Dict[str, Any]]:
    """Spread the validation sample across question types.

    Taking the first N would take only `pinpoint` answers, which are one or two
    claims long. Detecting an injected falsehood there is far easier than in a long
    architectural answer, so validating only on those would overstate the judge.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(row)

    picked: List[Dict[str, Any]] = []
    while len(picked) < sample and any(by_type.values()):
        for qtype in sorted(by_type):
            if by_type[qtype] and len(picked) < sample:
                picked.append(by_type[qtype].pop(0))
    return picked


def validate(judge: FaithfulnessJudge, rows, retriever, sample: int) -> None:
    """Inject a false claim into real answers; the judge must notice."""
    targets = stratified_sample(gradeable(rows), sample)
    detected = missed = 0
    baseline_flags = 0

    print(f"Validating the judge on {len(targets)} answers "
          f"(inject one unsupported claim, check it is caught)\n")

    # Question text is not stored in the answer rows, so pull it from the golden set.
    golden = load_golden_by_id()

    for index, row in enumerate(targets):
        question_text = golden[row["id"]]["question"]
        sources = retriever.retrieve(question_text, top_k=6)

        original = judge.judge(row["id"], question_text, row["answer"], sources)
        injected, keyword = PERTURBATIONS[index % len(PERTURBATIONS)]
        perturbed = judge.judge(
            row["id"], question_text, row["answer"] + " " + injected, sources
        )

        # Did the judge extract the injected claim AND mark it not-supported?
        matching = [c for c in perturbed.claims if keyword in c.claim.lower()]
        extracted = bool(matching)
        caught = any(c.verdict != "supported" for c in matching)

        detected += caught
        missed += not caught
        baseline_flags += original.unsupported + original.contradicted

        if caught:
            status = "caught"
        elif extracted:
            status = "WRONG"  # saw the claim, called it supported
        else:
            status = "DROPPED"  # never extracted the claim at all
        print(f"  {row['id']:<5} {row['type']:<14}{status:<8} "
              f"baseline_unsupported={original.unsupported}  "
              f"claims {original.total} -> {perturbed.total}")

    total = detected + missed
    print(f"\nInjected-claim detection: {detected}/{total} = {detected / total:.3f}" if total else "")
    print(f"Unsupported claims flagged on untouched answers: {baseline_flags} "
          f"(may be real unfaithfulness, or judge noise)")
    if total and detected / total < 0.8:
        print("\nJudge is unreliable: it misses injected falsehoods. Do not trust its scores.")
    else:
        print("\nJudge detects injected falsehoods; its scores are meaningful.")


def main():
    parser = argparse.ArgumentParser(description="Faithfulness judging")
    parser.add_argument("--label", default="step6", help="Answer run to grade")
    parser.add_argument("--repo", default="fastapi")
    parser.add_argument("--variant", default="astcode-cards")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--validate", action="store_true", help="Run the perturbation check")
    parser.add_argument("--sample", type=int, default=8, help="Answers used for validation")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    payload = load_answers(args.label)
    rows = payload["results"]
    retriever = build_retriever(args.repo, args.variant)
    judge = FaithfulnessJudge()

    if args.validate:
        validate(judge, rows, retriever, args.sample)
        return

    golden = load_golden_by_id()

    targets = gradeable(rows)
    if args.limit:
        targets = targets[: args.limit]

    results = []
    for index, row in enumerate(targets, start=1):
        question_text = golden[row["id"]]["question"]
        sources = retriever.retrieve(question_text, top_k=args.top_k)
        results.append(judge.judge(row["id"], question_text, row["answer"], sources))
        print(f"  judged {index}/{len(targets)}", end="\r", flush=True)
    print(" " * 40, end="\r")

    summary = aggregate(results)
    print(f"\n=== faithfulness: {args.label} (judge={judge.model}) ===")
    print(f"  answers graded          : {summary['answers_graded']}")
    print(f"  total claims            : {summary['total_claims']}")
    print(f"  supported               : {summary['supported']}")
    print(f"  unsupported             : {summary['unsupported']}")
    print(f"  contradicted            : {summary['contradicted']}")
    print(f"  claim faithfulness      : {summary['claim_faithfulness']:.3f}")
    print(f"  mean answer faithfulness: {summary['mean_answer_faithfulness']:.3f}")
    print(f"  fully clean answers     : {summary['clean_answer_rate']:.3f}")

    worst = sorted(
        (r for r in results if r.faithfulness is not None), key=lambda r: r.faithfulness
    )[:5]
    if worst:
        print("\nLeast faithful answers:")
        for result in worst:
            print(f"  {result.question_id}: {result.faithfulness:.2f} "
                  f"({result.unsupported} unsupported, {result.contradicted} contradicted)")

    out_path = config.EVAL_RESULTS_DIR / f"faithfulness-{args.label}.json"
    out_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "judge_model": judge.model,
                "generator_model": payload.get("generator_model"),
                "summary": summary,
                "results": [r.to_dict() for r in results],
            },
            indent=2,
        )
    )
    print(f"\nSaved: {out_path.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
