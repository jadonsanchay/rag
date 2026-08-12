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
import logging
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.auth import require_user
from api.auth import router as auth_router
from api.metrics import metrics
from api.ratelimit import client_ip, is_limited_path, limiter
from api.schemas import (
    AnswerOut,
    AskRequest,
    CitationOut,
    FileOut,
    SourceOut,
)
from api.conversations import router as conversations_router
from api.repos import router as repos_router
from api.sse import sse_comment, sse_event
from pipeline.registry import User
from pipeline import config
from pipeline.generator import AnswerGenerator
from pipeline.logging_config import configure_logging
from pipeline.manifests import load_manifest_for, manifest_path, repo_root_for
from pipeline.retriever import HybridRetriever
from pipeline.vector_store import collection_name_for_repo
from query import build_retriever

configure_logging()
logger = logging.getLogger("codebase_qa.api")

MANIFEST_DIR = config.DATA_DIR / "index_manifests"
PREVIEW_CHARS = 400
MAX_FILE_LINES = 2000

app = FastAPI(title="Codebase Q&A", version="0.12.0")

# The step 8 frontend runs on a separate dev-server origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()

    if is_limited_path(request.method, request.url.path):
        rejection = limiter.check(client_ip(request))
        if rejection:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "rate limited",
                extra={"method": request.method, "path": request.url.path, "duration_ms": round(duration_ms, 1)},
            )
            metrics.record_request(request.url.path, 429, duration_ms)
            return JSONResponse(status_code=429, content={"detail": rejection})

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "request failed",
            extra={"method": request.method, "path": request.url.path, "duration_ms": round(duration_ms, 1)},
        )
        metrics.record_request(request.url.path, 500, duration_ms)
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 1),
        },
    )
    metrics.record_request(request.url.path, response.status_code, duration_ms)
    return response


# /health and /metrics stay unprefixed (infra checks hit them directly); every
# route the frontend calls through api.ts lives under /api so one origin can
# serve both the SPA and the API in production (see the StaticFiles mount below).
app.include_router(auth_router, prefix="/api")
app.include_router(repos_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")


@app.on_event("startup")
def backfill_registry() -> None:
    """Import manifests written before the registry existed, so CLI-indexed
    repos appear alongside ones added through the API."""
    from pipeline.registry import import_manifests

    imported = import_manifests()
    if imported:
        logger.info("startup: imported repos from manifests", extra={"count": imported})


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


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


def embedding_for(repo: str, variant: str) -> tuple:
    """The provider/model a collection was built with.

    Collections are not interchangeable: one built with MiniLM holds 384-dim
    vectors and rejects a 1536-dim OpenAI query outright. The registry records
    what was used, with the manifest as a fallback for older indexes.
    """
    from pipeline.registry import get_registry

    entry = get_registry().find_repo(repo, variant)
    if entry and entry.embedding_model:
        return entry.embedding_provider, entry.embedding_model

    manifest = load_manifest_for(repo, variant)
    return manifest.get("embedding_provider"), manifest.get("embedding_model")


def get_retriever(repo: str, variant: str, mode: str) -> HybridRetriever:
    key = (repo, variant, mode)
    if key not in _retrievers:
        provider, model = embedding_for(repo, variant)
        _retrievers[key] = build_retriever(
            repo, variant, mode, provider=provider, model=model
        )
    return _retrievers[key]


def get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


def repo_root(repo: str, variant: Optional[str] = None) -> Path:
    """Where the indexed tree lives, per its manifest."""
    root = repo_root_for(repo, variant)
    if root is None:
        raise HTTPException(status_code=404, detail=f"Repo '{repo}' is not on disk")
    return root.resolve()


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


@app.get("/api/file", response_model=FileOut)
def read_file(
    repo: str = Query(default="fastapi"),
    variant: Optional[str] = Query(default=None),
    path: str = Query(..., min_length=1),
    start: int = Query(default=1, ge=1),
    end: Optional[int] = Query(default=None, ge=1),
    context: int = Query(default=0, ge=0, le=200),
) -> FileOut:
    """Return a slice of a file, for the source viewer behind a citation."""
    root = repo_root(repo, variant)
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
        logger.exception("retrieval failed", extra={"repo": request.repo})
        yield sse_event("error", {"message": "Retrieval failed", "detail": str(exc)})
        return

    retrieval_ms = int((time.perf_counter() - started) * 1000)

    if not results:
        metrics.record_answer(refused=True, retrieval_ms=retrieval_ms, generation_ms=0, invalid_citations=0)
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
        logger.exception("generation failed", extra={"repo": request.repo})
        yield sse_event("error", {"message": "Generation failed", "detail": str(exc)})
        return

    text = "".join(chunks)
    generation_ms = int((time.perf_counter() - generation_started) * 1000)

    # Verification needs the complete answer, so it runs after the token stream.
    answer = generator.finalize(
        text, results, repo_root=repo_root(request.repo, request.variant)
    )
    verification = answer.verification
    invalid_citations = len(verification.fabricated_indices) if verification else 0

    metrics.record_answer(
        refused=answer.refused,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        invalid_citations=invalid_citations,
    )
    logger.info(
        "answer",
        extra={
            "repo": request.repo,
            "refused": answer.refused,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "invalid_citations": invalid_citations,
        },
    )

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


@app.post("/api/ask")
async def ask(request: AskRequest, user: User = Depends(require_user)) -> StreamingResponse:
    """Stream an answer: trace, then tokens, then the verified result.

    Retrieval and generation are both blocking, so the synchronous generator is
    pumped on a worker thread — otherwise a single question would stall the event
    loop and block every other request.
    """
    # Validate before the stream opens: once a 200 + text/event-stream is
    # committed, a client-visible status code is no longer available.
    ensure_indexed(request.repo, request.variant)
    repo_root(request.repo, request.variant)
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


# Serves the built SPA from the same process/origin in production, so there is
# one Docker image, one deploy, and no CORS to configure. Registered last so
# every /api, /health, and /metrics route above still wins first — this catch-all
# only runs for GETs nothing else matched. Absent in local dev (no `web/dist`
# until `npm run build` has been run), where Vite's own dev server serves the UI.
#
# A plain StaticFiles mount isn't enough: the SPA now has a real client-side
# route (/app, via react-router) with no matching file on disk, so a direct
# load or refresh at that path must still get index.html — the router then
# takes over in the browser. Real files (JS/CSS bundles, favicon) are served
# as themselves; anything else falls back to index.html.
_frontend_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _frontend_dist.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str) -> FileResponse:
        candidate = (_frontend_dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(_frontend_dist):
            return FileResponse(candidate)
        return FileResponse(_frontend_dist / "index.html")
