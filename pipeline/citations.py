"""Citation parsing and verification.

A RAG answer is only useful if a reader can check it. Two things can go wrong that
look identical to the user:

  1. the model cites a source number that was never in the context (fabricated
     reference), and
  2. the model cites a real source whose line range does not exist in the file
     (stale or wrong span).

Both are checkable without an LLM, so they are checked mechanically here rather
than trusted.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass
class CitationCheck:
    index: int
    path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    exists: bool = False
    span_valid: bool = False
    content_matches: bool = False
    problem: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.exists and self.span_valid


@dataclass
class AnswerVerification:
    cited_indices: List[int] = field(default_factory=list)
    fabricated_indices: List[int] = field(default_factory=list)
    checks: List[CitationCheck] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_indices)

    @property
    def valid_citations(self) -> int:
        return sum(1 for check in self.checks if check.ok)

    @property
    def all_valid(self) -> bool:
        return not self.fabricated_indices and all(check.ok for check in self.checks)

    def summary(self) -> str:
        if not self.cited_indices:
            return "no citations"
        parts = [f"{self.valid_citations}/{len(self.checks)} citations verified"]
        if self.fabricated_indices:
            parts.append(f"fabricated: {self.fabricated_indices}")
        return ", ".join(parts)


def parse_markers(text: str) -> List[int]:
    """Extract [n] markers in first-appearance order."""
    return list(dict.fromkeys(int(match) for match in CITATION_MARKER.findall(text)))


def verify_span(
    metadata: Dict[str, Any], repo_root: Path, chunk_text: Optional[str] = None
) -> CitationCheck:
    """Confirm a chunk's cited location actually exists in the working tree."""
    check = CitationCheck(
        index=-1,
        path=metadata.get("path"),
        start_line=metadata.get("start_line"),
        end_line=metadata.get("end_line"),
    )

    if not check.path:
        check.problem = "no path in metadata"
        return check

    # Package cards point at a directory, not a file.
    if check.path.endswith("/"):
        directory = repo_root / check.path
        check.exists = directory.is_dir()
        check.span_valid = check.exists
        if not check.exists:
            check.problem = "directory not found"
        return check

    file_path = repo_root / check.path
    if not file_path.is_file():
        check.problem = "file not found"
        return check
    check.exists = True

    if check.start_line is None:
        # Prose chunks from the baseline splitter carry no line range; the file
        # existing is all that can be verified.
        check.span_valid = True
        return check

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not (1 <= check.start_line <= len(lines)):
        check.problem = f"start_line {check.start_line} outside file ({len(lines)} lines)"
        return check
    if check.end_line and check.end_line > len(lines):
        check.problem = f"end_line {check.end_line} outside file ({len(lines)} lines)"
        return check
    check.span_valid = True

    if chunk_text:
        cited = {line.strip() for line in lines[check.start_line - 1 : check.end_line or check.start_line]}
        header_lines = metadata.get("header_lines", 0) or 0
        body = [l.strip() for l in chunk_text.splitlines()[header_lines:] if l.strip()]
        check.content_matches = bool(body) and body[-1] in cited

    return check


def verify_answer(
    answer: str, sources: Sequence[Dict[str, Any]], repo_root: Path
) -> AnswerVerification:
    """Check every [n] marker in an answer against the sources it was given."""
    verification = AnswerVerification(cited_indices=parse_markers(answer))

    for index in verification.cited_indices:
        if not (1 <= index <= len(sources)):
            verification.fabricated_indices.append(index)
            continue
        source = sources[index - 1]
        check = verify_span(source["metadata"], repo_root, source.get("content"))
        check.index = index
        verification.checks.append(check)

    return verification
