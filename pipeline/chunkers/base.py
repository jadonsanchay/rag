from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from langchain_core.documents import Document

MAX_HEADER_CHARS = 300


@dataclass
class Chunk:
    """A retrievable unit of a file, with the location needed for citations."""

    text: str
    path: str
    start_line: int
    end_line: int
    kind: str  # function | class_skeleton | module | text
    language: str
    symbol: Optional[str] = None
    parent_symbol: Optional[str] = None
    part: Optional[int] = None  # set when an oversized chunk was windowed
    part_count: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_symbol(self) -> Optional[str]:
        if self.symbol and self.parent_symbol:
            return f"{self.parent_symbol}.{self.symbol}"
        return self.symbol

    def to_document(self, base_metadata: Optional[Dict[str, Any]] = None) -> Document:
        metadata: Dict[str, Any] = dict(base_metadata or {})
        metadata.update(
            {
                "path": self.path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "kind": self.kind,
                "language": self.language,
                "symbol": self.symbol,
                "parent_symbol": self.parent_symbol,
                "qualified_symbol": self.qualified_symbol,
                "part": self.part,
                "part_count": self.part_count,
            }
        )
        metadata.update(self.extra)
        # Chroma rejects None metadata values.
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return Document(page_content=self.text, metadata=metadata)


class Chunker(Protocol):
    """Language-specific splitters implement this.

    Two backends exist: the stdlib `ast` chunker for Python, and a tree-sitter
    chunker for everything else. Python deliberately keeps its own implementation
    — `ast` understands docstrings and decorators natively, and its results are
    already measured — so the interface exists to let the better parser win per
    language rather than to force one code path.
    """

    language: str

    def chunk(self, source: str, path: str) -> List[Chunk]: ...


def emit_windowed(
    text: str,
    path: str,
    start_line: int,
    end_line: int,
    kind: str,
    language: str,
    header: str,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    symbol: Optional[str] = None,
    parent_symbol: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """One chunk if it fits the token budget, otherwise windowed parts that each
    restate `header` so every piece stays self-identifying.

    Shared by both chunker backends: the windowing rules (per-line token costs,
    bounded headers, header_lines bookkeeping for citation verification) are
    language-independent and were expensive to get right once.
    """
    base_extra = dict(extra or {})

    if count_tokens(text) <= max_tokens:
        chunk = Chunk(
            text=text,
            path=path,
            start_line=start_line,
            end_line=end_line,
            kind=kind,
            language=language,
            symbol=symbol,
            parent_symbol=parent_symbol,
            extra=base_extra,
        )
        return [chunk]

    lines = text.splitlines()
    # Bound the header so it can never dominate the part it labels.
    header = header[:MAX_HEADER_CHARS]
    header_cost = min(count_tokens(header), max_tokens // 4)
    budget = max(max_tokens - header_cost, max_tokens // 4)

    # Per-line costs, summed. Re-tokenizing the growing window is quadratic;
    # summing per-line counts slightly over-estimates, erring on the safe side.
    line_costs = [count_tokens(line) + 1 for line in lines]

    windows: List[Tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        cursor = index
        running = 0
        while cursor < len(lines):
            if cursor > index and running + line_costs[cursor] > budget:
                break
            running += line_costs[cursor]
            cursor += 1
        windows.append((index, cursor, "\n".join(lines[index:cursor])))
        if cursor >= len(lines):
            break
        index = cursor

    header_lines = len(header.splitlines())
    chunks: List[Chunk] = []
    for part_number, (offset, cursor, body) in enumerate(windows, start=1):
        is_continuation = part_number > 1
        prefix = f"{header}\n" if is_continuation else ""
        part_extra = dict(base_extra)
        # Continuation parts carry a synthetic header that is NOT present at
        # start_line; a source viewer must skip those lines when mapping back.
        part_extra["header_lines"] = (
            base_extra.get("header_lines", 0) or 0
        ) + (header_lines if is_continuation else 0)

        chunks.append(
            Chunk(
                text=f"{prefix}{body}".strip(),
                path=path,
                start_line=start_line + offset,
                end_line=start_line + cursor - 1,
                kind=kind,
                language=language,
                symbol=symbol,
                parent_symbol=parent_symbol,
                part=part_number,
                part_count=len(windows),
                extra=part_extra,
            )
        )
    return chunks
