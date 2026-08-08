"""Multi-turn chat endpoints.

The stream gains one frame over `/ask`: a `rewrite` event carrying the standalone
query a follow-up was condensed into. That is the step most likely to surprise a
user ("why did it search for *that*?"), so it is shown rather than hidden.

    event: rewrite   original + rewritten query (only when rewriting happened)
    event: trace     retrieved sources
    event: token     answer deltas
    event: done      answer, citations, message id
"""

import time
from typing import Iterator, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import (
    AnswerOut,
    ChatRequest,
    CitationOut,
    ConversationOut,
    MessageOut,
    NewConversationRequest,
)
from api.sse import sse_comment, sse_event
from pipeline.conversation import QueryRewriter
from pipeline.manifests import repo_root_for
from pipeline.registry import get_registry

router = APIRouter(prefix="/conversations", tags=["conversations"])

_rewriter: Optional[QueryRewriter] = None


def get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


@router.post("", response_model=ConversationOut)
def create_conversation(request: NewConversationRequest) -> ConversationOut:
    registry = get_registry()
    repo = registry.get_repo(request.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Unknown repo")
    if not repo.ready:
        raise HTTPException(
            status_code=409,
            detail=f"Repo is not ready (status: {repo.status})",
        )

    conversation_id = registry.create_conversation(request.repo_id)
    return ConversationOut(id=conversation_id, repo_id=request.repo_id)


@router.get("/{conversation_id}/messages", response_model=List[MessageOut])
def list_messages(conversation_id: str) -> List[MessageOut]:
    registry = get_registry()
    if registry.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return [MessageOut(**message) for message in registry.messages(conversation_id)]


def chat_stream(conversation_id: str, request: ChatRequest) -> Iterator[str]:
    from api.main import get_generator, get_retriever, to_source_out

    registry = get_registry()
    conversation = registry.get_conversation(conversation_id)
    repo = registry.get_repo(conversation["repo_id"])
    started = time.perf_counter()

    yield sse_comment("stream open")

    # Rewrite before retrieval. A follow-up like "and where is that called from?"
    # retrieves nothing useful as written.
    history = registry.history_turns(conversation_id)
    try:
        rewrite = get_rewriter().rewrite(request.question, history)
    except Exception as exc:  # noqa: BLE001 - degrade to the raw question
        yield sse_event("error", {"message": "Rewrite failed", "detail": str(exc)})
        return

    if rewrite.changed:
        yield sse_event(
            "rewrite",
            {
                "original": rewrite.original,
                "query": rewrite.query,
                "reason": rewrite.reason,
            },
        )

    registry.add_message(
        conversation_id,
        "user",
        request.question,
        rewritten_query=rewrite.query if rewrite.changed else None,
    )

    try:
        retriever = get_retriever(repo.name, repo.variant, request.mode)
        results = retriever.retrieve(rewrite.query, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        yield sse_event("error", {"message": "Retrieval failed", "detail": str(exc)})
        return

    retrieval_ms = int((time.perf_counter() - started) * 1000)
    sources = [to_source_out(i, r).model_dump() for i, r in enumerate(results, start=1)]
    yield sse_event(
        "trace",
        {"sources": sources, "retrieval_ms": retrieval_ms, "query": rewrite.query},
    )

    if not results:
        registry.add_message(conversation_id, "assistant", "", trace=[])
        yield sse_event(
            "done",
            AnswerOut(answer="", refused=True, timing_ms={"retrieval": retrieval_ms}).model_dump(),
        )
        return

    generation_started = time.perf_counter()
    generator = get_generator()
    chunks: List[str] = []
    try:
        # History is passed so the answer can refer back naturally, but only this
        # turn's sources are in scope for citation numbering.
        for delta in generator.stream(rewrite.query, results, history=history):
            chunks.append(delta)
            yield sse_event("token", {"text": delta})
    except Exception as exc:  # noqa: BLE001
        yield sse_event("error", {"message": "Generation failed", "detail": str(exc)})
        return

    text = "".join(chunks)
    answer = generator.finalize(
        text, results, repo_root=repo_root_for(repo.name, repo.variant)
    )
    verification = answer.verification

    message_id = registry.add_message(
        conversation_id,
        "assistant",
        answer.text,
        trace=[{"index": s["index"], "path": s["path"]} for s in sources],
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
        timing_ms={
            "retrieval": retrieval_ms,
            "generation": int((time.perf_counter() - generation_started) * 1000),
        },
    ).model_dump()
    payload["message_id"] = message_id
    yield sse_event("done", payload)


@router.post("/{conversation_id}/messages")
async def send_message(conversation_id: str, request: ChatRequest) -> StreamingResponse:
    registry = get_registry()
    conversation = registry.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")

    repo = registry.get_repo(conversation["repo_id"])
    if repo is None or not repo.ready:
        raise HTTPException(status_code=409, detail="Repo is not ready")

    frames = chat_stream(conversation_id, request)

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
            "X-Accel-Buffering": "no",
        },
    )
