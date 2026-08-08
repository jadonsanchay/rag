"""HTTP surface for the codebase Q&A pipeline.

The streaming contract is the point of this layer. `/ask` emits, in order:

    event: trace   the retrieved sources, before a single token is generated
    event: token   answer deltas as they arrive
    event: done    the assembled answer plus verified citations

Retrieval takes roughly a second (one embedding call plus two index lookups) and
generation several more. Sending the trace first means the UI can render sources
and let the user start reading code while the answer is still being written,
instead of showing a spinner for the whole wall-clock time.
"""

import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import (
    AnswerOut,
    AskRequest,
    CitationOut,
    FileOut,
    RepoOut,
    SourceOut,
)
from api.sse import sse_comment, sse_event
from pipeline import config
from pipeline.generator import AnswerGenerator
from pipeline.retriever import HybridRetriever
from pipeline.vector_store import collection_name_for_repo
from query import build_retriever

MANIFEST_DIR = config.DATA_DIR / "index_manifests"
PREVIEW_CHARS = 400
MAX_FILE_LINES = 2000

app = FastAPI(title="Codebase Q&A", version="0.7.0")

# The step 8 frontend runs on a separate dev-server origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Building a retriever opens a Chroma client and a SQLite connection, so they are
# reused across requests rather than rebuilt per question.
_retrievers: Dict[Tuple[str, str, str], HybridRetriever] = {}
_generator: Optional[AnswerGenerator] = None


def ensure_indexed(repo: str, variant: str) -> None:
    """Fail loudly for an unindexed collection.

    Without this, an unknown variant resolves to an empty Chroma collection, so
    retrieval returns nothing and the answer comes back as a refusal — telling the
    user their question was unanswerable when the truth is the repo was never
    indexed. Two very different problems should not look identical.
    """
    collection = collection_name_for_repo(repo, variant)
    if not (MANIFEST_DIR / f"{collection}.json").exists():
        available = sorted(p.stem for p in MANIFEST_DIR.glob("*.json"))
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No index for repo='{repo}' variant='{variant}'",
                "available_collections": available,
            },
        )


def get_retriever(repo: str, variant: str, mode: str) -> HybridRetriever:
    key = (repo, variant, mode)
    if key not in _retrievers:
        _retrievers[key] = build_retriever(repo, variant, mode)
    return _retrievers[key]


def get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


def repo_root(repo: str) -> Path:
    root = (config.REPOS_DIR / repo).resolve()
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Repo '{repo}' is not on disk")
    return root


def to_source_out(index: int, result: dict) -> SourceOut:
    metadata = result["metadata"]
    return SourceOut(
        index=index,
        path=metadata.get("path", "unknown"),
        start_line=metadata.get("start_line"),
        end_line=metadata.get("end_line"),
        symbol=metadata.get("qualified_symbol") or metadata.get("symbol"),
        kind=metadata.get("kind"),
        language=metadata.get("language"),
        score=round(float(result.get("score") or 0.0), 6),
        ranks=result.get("ranks") or {},
        retrievers=result.get("sources") or [],
        preview=result["content"][:PREVIEW_CHARS],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_repos": len(list(MANIFEST_DIR.glob("*.json")))}


@app.get("/repos", response_model=List[RepoOut])
def list_repos() -> List[RepoOut]:
    """Indexed collections, read from the manifests written by index_repo.py."""
    repos = []
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        repos.append(
            RepoOut(
                repo=manifest.get("repo", path.stem),
                variant=manifest.get("variant", ""),
                collection=manifest.get("collection", path.stem),
                chunks=manifest.get("chunks", 0),
                files_indexed=manifest.get("files_indexed"),
                commit_sha=manifest.get("commit_sha"),
                embedding_model=manifest.get("embedding_model"),
            )
        )
    return repos


@app.get("/file", response_model=FileOut)
def read_file(
    repo: str = Query(default="fastapi"),
    path: str = Query(..., min_length=1),
    start: int = Query(default=1, ge=1),
    end: Optional[int] = Query(default=None, ge=1),
    context: int = Query(default=0, ge=0, le=200),
) -> FileOut:
    """Return a slice of a file, for the source viewer behind a citation."""
    root = repo_root(repo)
    target = (root / path).resolve()

    # Path traversal guard: a citation path is user-controllable input.
    if not target.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Path escapes the repository")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such file: {path}")

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    first = max(1, start - context)
    last = min(total, (end or start) + context)
    if first > total:
        raise HTTPException(status_code=416, detail=f"start {start} beyond EOF ({total})")
    if last - first + 1 > MAX_FILE_LINES:
        last = first + MAX_FILE_LINES - 1

    return FileOut(
        path=path,
        start_line=first,
        end_line=last,
        total_lines=total,
        language=config.LANGUAGE_BY_EXTENSION.get(target.suffix.lower()),
        content="\n".join(lines[first - 1 : last]),
    )


def ask_stream(request: AskRequest) -> Iterator[str]:
    """Generate the SSE frames for one question."""
    started = time.perf_counter()
    yield sse_comment("stream open")

    try:
        retriever = get_retriever(request.repo, request.variant, request.mode)
        results = retriever.retrieve(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001 - must reach the client as an event
        yield sse_event("error", {"message": "Retrieval failed", "detail": str(exc)})
        return

    retrieval_ms = int((time.perf_counter() - started) * 1000)

    if not results:
        yield sse_event("trace", {"sources": [], "retrieval_ms": retrieval_ms})
        yield sse_event(
            "done",
            AnswerOut(
                answer="",
                refused=True,
                timing_ms={"retrieval": retrieval_ms, "generation": 0},
            ).model_dump(),
        )
        return

    sources = [to_source_out(i, r).model_dump() for i, r in enumerate(results, start=1)]
    # The whole point: sources reach the UI before generation starts.
    yield sse_event("trace", {"sources": sources, "retrieval_ms": retrieval_ms})

    generation_started = time.perf_counter()
    generator = get_generator()
    chunks: List[str] = []
    try:
        for delta in generator.stream(request.question, results):
            chunks.append(delta)
            yield sse_event("token", {"text": delta})
    except Exception as exc:  # noqa: BLE001
        yield sse_event("error", {"message": "Generation failed", "detail": str(exc)})
        return

    text = "".join(chunks)
    generation_ms = int((time.perf_counter() - generation_started) * 1000)

    # Verification needs the complete answer, so it runs after the token stream.
    answer = generator.finalize(text, results, repo_root=repo_root(request.repo))
    verification = answer.verification

    payload = AnswerOut(
        answer=answer.text,
        refused=answer.refused,
        cited_indices=answer.cited_indices,
        citations=[
            CitationOut(
                index=check.index,
                path=check.path,
                start_line=check.start_line,
                end_line=check.end_line,
                valid=check.ok,
                problem=check.problem,
            )
            for check in (verification.checks if verification else [])
        ],
        fabricated_indices=verification.fabricated_indices if verification else [],
        citation_summary=verification.summary() if verification else "",
        timing_ms={"retrieval": retrieval_ms, "generation": generation_ms},
    )
    yield sse_event("done", payload.model_dump())


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    """Stream an answer: trace, then tokens, then the verified result.

    Retrieval and generation are both blocking, so the synchronous generator is
    pumped on a worker thread — otherwise a single question would stall the event
    loop and block every other request.
    """
    # Validate before the stream opens: once a 200 + text/event-stream is
    # committed, a client-visible status code is no longer available.
    ensure_indexed(request.repo, request.variant)
    repo_root(request.repo)
    frames = ask_stream(request)

    async def pump():
        sentinel = object()
        while True:
            frame = await run_in_threadpool(next, frames, sentinel)
            if frame is sentinel:
                break
            yield frame

    return StreamingResponse(
        pump(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx buffering the stream
        },
    )
