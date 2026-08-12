"""Repo lifecycle endpoints: add by URL, poll status, re-index, delete.

Repos are per-user visible but the underlying index is shared: if another
user already has a `ready` row for the same collection (same GitHub repo +
variant), adding it here attaches a new row to the existing index instead of
re-cloning and re-embedding — see `add_repo`.
"""

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.auth import require_user
from api.schemas import AddRepoRequest, RepoStatusOut
from pipeline.ingest_job import (
    IngestError,
    delete_repo_data,
    parse_github_url,
    run_ingest,
)
from pipeline.registry import CLONING, INDEXING, QUEUED, Repo, User, get_registry
from pipeline.vector_store import collection_name_for_repo

router = APIRouter(prefix="/repos", tags=["repos"])

IN_PROGRESS = {QUEUED, CLONING, INDEXING}


def to_status(repo: Repo) -> RepoStatusOut:
    return RepoStatusOut(
        id=repo.id,
        name=repo.name,
        url=repo.url,
        variant=repo.variant,
        collection=repo.collection,
        status=repo.status,
        stage=repo.stage,
        error=repo.error,
        commit_sha=repo.commit_sha,
        files_indexed=repo.files_indexed,
        chunks=repo.chunks,
        languages=repo.languages,
        indexed_at=repo.indexed_at,
        ready=repo.ready,
    )


def require_repo(repo_id: str, user: User) -> Repo:
    """404, not 403, on a repo id that exists but belongs to someone else —
    don't confirm it exists to a non-owner."""
    repo = get_registry().get_repo(repo_id)
    if repo is None or repo.user_id != user.id:
        raise HTTPException(status_code=404, detail=f"No repo with id '{repo_id}'")
    return repo


@router.get("", response_model=List[RepoStatusOut])
def list_repos(user: User = Depends(require_user)) -> List[RepoStatusOut]:
    return [to_status(repo) for repo in get_registry().list_repos(user.id)]


@router.post("", response_model=RepoStatusOut, status_code=202)
def add_repo(
    request: AddRepoRequest,
    background: BackgroundTasks,
    user: User = Depends(require_user),
) -> RepoStatusOut:
    """Accept a GitHub URL and start indexing in the background — unless
    someone already has this exact repo+variant `ready`, in which case this
    user gets their own row over the existing index for free.

    Returns 202 immediately when a job is actually started: cloning and
    embedding take minutes, so the client polls GET /repos/{id} for the stage
    rather than holding a request open.
    """
    try:
        parsed = parse_github_url(request.url)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry = get_registry()
    collection = collection_name_for_repo(parsed.slug, request.variant)

    mine = registry.find_user_repo_by_collection(user.id, collection)
    if mine:
        if mine.status in IN_PROGRESS:
            # Idempotent: re-submitting a repo already being indexed for you
            # is not an error.
            return to_status(mine)
        if mine.status == "failed":
            registry.update_repo(mine.id, status=QUEUED, stage="queued", error=None)
            background.add_task(run_ingest, mine.id, request.url)
            return to_status(registry.get_repo(mine.id))
        return to_status(mine)  # already ready

    ready_elsewhere = registry.find_ready_repo_by_collection(collection)
    if ready_elsewhere:
        # Someone else already built this index — reuse it, no clone/embed spend.
        repo = registry.attach_existing_repo(ready_elsewhere, user_id=user.id)
        return to_status(repo)

    repo = registry.create_repo(
        name=parsed.slug,
        variant=request.variant,
        collection=collection,
        url=request.url,
        status=QUEUED,
        user_id=user.id,
    )
    background.add_task(run_ingest, repo.id, request.url)
    return to_status(repo)


@router.get("/{repo_id}", response_model=RepoStatusOut)
def get_repo(repo_id: str, user: User = Depends(require_user)) -> RepoStatusOut:
    return to_status(require_repo(repo_id, user))


@router.post("/{repo_id}/reindex", response_model=RepoStatusOut, status_code=202)
def reindex(
    repo_id: str, background: BackgroundTasks, user: User = Depends(require_user)
) -> RepoStatusOut:
    """Re-index in place. Re-clones when the repo came from a URL, so a stale
    working tree is refreshed rather than re-embedded as-is.

    Rebuilds the shared index — every other user's row pointing at this same
    collection picks up the refreshed stats too (see `run_ingest`)."""
    repo = require_repo(repo_id, user)
    registry = get_registry()
    registry.update_repo(repo_id, status=QUEUED, stage="queued", error=None)
    background.add_task(run_ingest, repo_id, repo.url)
    return to_status(registry.get_repo(repo_id))


@router.delete("/{repo_id}", status_code=204)
def delete_repo(
    repo_id: str, remove_clone: bool = True, user: User = Depends(require_user)
) -> None:
    """Drop this user's registry row. The underlying index (Chroma collection,
    lexical index, cloned tree) is only torn down if no other user's row still
    references the same collection — otherwise it would pull the index out
    from under someone else's repo list."""
    repo = require_repo(repo_id, user)
    registry = get_registry()
    teardown_index = not registry.collection_in_use(repo.collection, excluding_repo_id=repo_id)
    delete_repo_data(repo_id, remove_clone=remove_clone, teardown_index=teardown_index)
