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


# --- steps 10-12: repo lifecycle and conversations -------------------------


class AddRepoRequest(BaseModel):
    url: str = Field(min_length=8, max_length=300)
    variant: str = Field(default="main", pattern=r"^[A-Za-z0-9_-]{1,32}$")


class RepoStatusOut(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    variant: str
    collection: str
    status: str
    stage: Optional[str] = None
    error: Optional[str] = None
    commit_sha: Optional[str] = None
    files_indexed: int = 0
    chunks: int = 0
    languages: Dict[str, int] = Field(default_factory=dict)
    indexed_at: Optional[float] = None
    ready: bool = False


class NewConversationRequest(BaseModel):
    repo_id: str


class ConversationOut(BaseModel):
    id: str
    repo_id: str
    title: Optional[str] = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    rewritten_query: Optional[str] = None
    trace: Optional[Any] = None
    created_at: float


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: str = Field(default="hybrid", pattern="^(semantic|lexical|hybrid)$")
    top_k: int = Field(default=6, ge=1, le=20)


# --- auth --------------------------------------------------------------


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    # bcrypt silently truncates beyond 72 bytes — cap here so a longer
    # password doesn't quietly behave differently on hash vs. later verify.
    password: str = Field(min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=72)


class UserOut(BaseModel):
    id: str
    email: str
