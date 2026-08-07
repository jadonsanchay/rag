from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from langchain_core.documents import Document


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
    """Language-specific splitters implement this. Step 9 adds tree-sitter
    backed implementations behind the same interface."""

    language: str

    def chunk(self, source: str, path: str) -> List[Chunk]: ...
