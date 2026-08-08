from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    repo: str = "fastapi"
    variant: str = "astcode-cards"
    mode: str = Field(default="hybrid", pattern="^(semantic|lexical|hybrid)$")
    top_k: int = Field(default=6, ge=1, le=20)


class SourceOut(BaseModel):
    """One retrieved chunk, as the UI needs it: where it came from, why it ranked,
    and enough location detail to open the file at the right place."""

    index: int
    path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    symbol: Optional[str] = None
    kind: Optional[str] = None
    language: Optional[str] = None
    score: float
    ranks: Dict[str, int] = Field(default_factory=dict)
    retrievers: List[str] = Field(default_factory=list)
    preview: str


class RepoOut(BaseModel):
    repo: str
    variant: str
    collection: str
    chunks: int
    files_indexed: Optional[int] = None
    commit_sha: Optional[str] = None
    embedding_model: Optional[str] = None
    indexed_at: Optional[str] = None


class FileOut(BaseModel):
    path: str
    start_line: int
    end_line: int
    total_lines: int
    language: Optional[str] = None
    content: str


class CitationOut(BaseModel):
    index: int
    path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    valid: bool
    problem: Optional[str] = None


class AnswerOut(BaseModel):
    answer: str
    refused: bool
    cited_indices: List[int] = Field(default_factory=list)
    citations: List[CitationOut] = Field(default_factory=list)
    fabricated_indices: List[int] = Field(default_factory=list)
    citation_summary: str = ""
    timing_ms: Dict[str, int] = Field(default_factory=dict)


class ErrorOut(BaseModel):
    message: str
    detail: Optional[Any] = None
