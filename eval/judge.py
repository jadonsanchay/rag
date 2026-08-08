"""Claim-level faithfulness judging.

`citations.py` proves a cited span *exists*. It cannot tell whether the claim
matches what is at that span — an answer can cite `routing.py:593-618` and assert
"APIRouter manages database connections" and pass every mechanical check. Closing
that gap needs a model, so this module adds one.

Two deliberate choices:

  * The judge grades **faithfulness to the provided sources**, not correctness in
    the abstract. Asking "is this true about FastAPI?" invites the judge to answer
    from its own knowledge of a popular library, which measures the judge rather
    than the system.
  * The judge model is **different from and stronger than the generator**
    (gpt-4o vs gpt-4o-mini). A model grading its own output shares its blind spots.

The judge is itself validated by perturbation — see validate_judge.py.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

JUDGE_MODEL = "gpt-4o"

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"

JUDGE_PROMPT = """You are grading whether an answer is faithful to the sources it \
was given. You are NOT grading whether the answer is true in general — only \
whether the numbered sources support it.

Steps:
1. Split the answer into atomic factual claims. Ignore hedges, restatements of the
   question, and pure connective prose.
2. Judge each claim against the sources ONLY:
   - "supported": a source states or directly implies it.
   - "unsupported": no source establishes it, even if it sounds plausible or you
     believe it is true from your own knowledge.
   - "contradicted": a source states something incompatible with it.

Do not use knowledge outside the sources. A claim you know to be true about this \
library is still "unsupported" if the sources do not establish it.

Return JSON only:
{"claims": [{"claim": "<short paraphrase>", "verdict": "supported|unsupported|contradicted", "source": <number or null>}]}"""


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    source: Optional[int] = None


@dataclass
class JudgeResult:
    question_id: str
    claims: List[ClaimVerdict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == SUPPORTED)

    @property
    def unsupported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == UNSUPPORTED)

    @property
    def contradicted(self) -> int:
        return sum(1 for c in self.claims if c.verdict == CONTRADICTED)

    @property
    def faithfulness(self) -> Optional[float]:
        """Fraction of claims the sources support. None when there are no claims."""
        return round(self.supported / self.total, 3) if self.total else None

    @property
    def clean(self) -> bool:
        return self.total > 0 and self.unsupported == 0 and self.contradicted == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "total_claims": self.total,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "contradicted": self.contradicted,
            "faithfulness": self.faithfulness,
            "clean": self.clean,
            "error": self.error,
            "claims": [
                {"claim": c.claim, "verdict": c.verdict, "source": c.source}
                for c in self.claims
            ],
        }


class FaithfulnessJudge:
    def __init__(self, model: str = JUDGE_MODEL):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self.model = model
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def format_sources(sources: Sequence[Dict[str, Any]]) -> str:
        blocks = []
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            path = metadata.get("path", "unknown")
            start, end = metadata.get("start_line"), metadata.get("end_line")
            location = f"{path}:{start}-{end}" if start else path
            blocks.append(f"[{index}] {location}\n{source.get('content', '')}")
        return "\n\n".join(blocks)

    def judge(
        self,
        question_id: str,
        question: str,
        answer: str,
        sources: Sequence[Dict[str, Any]],
    ) -> JudgeResult:
        prompt = (
            f"Question: {question}\n\n"
            f"Sources:\n{self.format_sources(sources)}\n\n"
            f"Answer to grade:\n{answer}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            payload = json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001 - judging must not abort a run
            return JudgeResult(question_id=question_id, error=str(exc))

        claims = []
        for raw in payload.get("claims", []):
            verdict = str(raw.get("verdict", "")).lower().strip()
            if verdict not in {SUPPORTED, UNSUPPORTED, CONTRADICTED}:
                verdict = UNSUPPORTED
            claims.append(
                ClaimVerdict(
                    claim=str(raw.get("claim", ""))[:300],
                    verdict=verdict,
                    source=raw.get("source") if isinstance(raw.get("source"), int) else None,
                )
            )

        return JudgeResult(question_id=question_id, claims=claims)


def aggregate(results: Sequence[JudgeResult]) -> Dict[str, Any]:
    graded = [r for r in results if r.total and not r.error]
    total_claims = sum(r.total for r in graded)
    supported = sum(r.supported for r in graded)
    unsupported = sum(r.unsupported for r in graded)
    contradicted = sum(r.contradicted for r in graded)

    per_answer = [r.faithfulness for r in graded if r.faithfulness is not None]
    return {
        "answers_graded": len(graded),
        "answers_skipped": len(results) - len(graded),
        "total_claims": total_claims,
        "supported": supported,
        "unsupported": unsupported,
        "contradicted": contradicted,
        # Claim-weighted: the share of all claims the sources back.
        "claim_faithfulness": round(supported / total_claims, 3) if total_claims else 0.0,
        # Answer-weighted: mean per-answer faithfulness, so a long answer does not
        # dominate a short one.
        "mean_answer_faithfulness": round(sum(per_answer) / len(per_answer), 3)
        if per_answer
        else 0.0,
        "clean_answer_rate": round(
            sum(1 for r in graded if r.clean) / len(graded), 3
        )
        if graded
        else 0.0,
    }
