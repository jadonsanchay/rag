"""Multi-turn support: rewriting follow-ups into standalone retrieval queries.

"And where is that called from?" is unretrievable as written. Embedded on its own it
matches nothing in particular, and BM25 sees only stopwords. The prior turns hold
the referent, so a follow-up must be condensed into a self-contained query *before*
it reaches either index.

This is the single step that separates multi-turn feeling broken from feeling
obvious, and it is cheap: one small-model call, skipped entirely when the question
already stands alone.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from . import config

load_dotenv()

REWRITE_MODEL = "gpt-4o-mini"
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 2400

REWRITE_PROMPT = """Rewrite the user's latest question so it can be understood \
without the conversation, for searching a codebase.

Rules:
- Resolve pronouns and references ("it", "that function", "there") using the history.
- Keep the user's own identifiers and terminology; do not invent symbol names.
- Keep it one short question. Do not answer it.
- If the question already stands alone, return it unchanged.

Return only the rewritten question."""

# Markers that a question depends on earlier turns. Cheap pre-filter so a
# self-contained question never pays for a rewrite call.
DEPENDENT_PATTERNS = re.compile(
    r"\b(it|its|it's|that|this|these|those|there|then|they|them|their|he|she|"
    r"same|above|previous|instead|also|too|as well)\b",
    re.IGNORECASE,
)
FOLLOW_UP_OPENERS = re.compile(
    r"^\s*(and|but|so|what about|how about|why|ok|okay|then|also)\b", re.IGNORECASE
)


@dataclass
class RewriteResult:
    query: str
    original: str
    rewritten: bool
    reason: str

    @property
    def changed(self) -> bool:
        return self.rewritten and self.query.strip() != self.original.strip()


def looks_dependent(question: str) -> bool:
    """Whether a question plausibly refers to earlier turns."""
    stripped = question.strip()
    if FOLLOW_UP_OPENERS.match(stripped):
        return True
    if DEPENDENT_PATTERNS.search(stripped):
        return True
    # Very short questions are usually elliptical ("where is it defined?").
    return len(stripped.split()) <= 4


def format_history(turns: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    budget = MAX_HISTORY_CHARS
    # Newest turns matter most for resolving a reference, so fill from the end.
    for turn in reversed(list(turns)[-MAX_HISTORY_TURNS:]):
        speaker = "User" if turn.get("role") == "user" else "Assistant"
        content = " ".join((turn.get("content") or "").split())[:600]
        entry = f"{speaker}: {content}"
        if len(entry) > budget:
            break
        lines.append(entry)
        budget -= len(entry)
    return "\n".join(reversed(lines))


class QueryRewriter:
    def __init__(self, model: str = REWRITE_MODEL):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def rewrite(
        self, question: str, history: Sequence[Dict[str, str]]
    ) -> RewriteResult:
        if not history:
            return RewriteResult(question, question, False, "no history")
        if not looks_dependent(question):
            return RewriteResult(question, question, False, "self-contained")

        prompt = (
            f"Conversation so far:\n{format_history(history)}\n\n"
            f"Latest question: {question}\n\nRewritten question:"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=80,
            )
            rewritten = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - degrade to the raw question
            return RewriteResult(question, question, False, f"rewrite failed: {exc}")

        rewritten = rewritten.strip().strip('"')
        if not rewritten or len(rewritten) > 400:
            return RewriteResult(question, question, False, "rewrite unusable")
        return RewriteResult(rewritten, question, True, "rewritten from history")


def build_chat_messages(
    question: str,
    context: str,
    history: Sequence[Dict[str, str]],
    system_prompt: str,
    max_history_turns: int = 4,
) -> List[Dict[str, str]]:
    """Assemble the generation messages for a conversational turn.

    History is included so the answer can refer back naturally ("as shown above"),
    but it is bounded and carries no retrieved context of its own: only the current
    turn's sources are in play, which keeps the model from citing chunk numbers
    from a previous turn that no longer exist.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for turn in list(history)[-max_history_turns:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        content = " ".join((turn.get("content") or "").split())[:800]
        if content:
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Sources for this question only — cite by these numbers:\n{context}"
                f"\n\nQuestion: {question}\n\nAnswer:"
            ),
        }
    )
    return messages
