"""Repo lifecycle endpoints: add by URL, poll status, re-index, delete."""

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import AddRepoRequest, RepoStatusOut
from pipeline.ingest_job import (
    IngestError,
    delete_repo_data,
    parse_github_url,
    run_ingest,
)
from pipeline.registry import QUEUED, Repo, get_registry
from pipeline.vector_store import collection_name_for_repo

router = APIRouter(prefix="/repos", tags=["repos"])


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


def require_repo(repo_id: str) -> Repo:
    repo = get_registry().get_repo(repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"No repo with id '{repo_id}'")
    return repo


@router.get("", response_model=List[RepoStatusOut])
def list_repos() -> List[RepoStatusOut]:
    return [to_status(repo) for repo in get_registry().list_repos()]


@router.post("", response_model=RepoStatusOut, status_code=202)
def add_repo(request: AddRepoRequest, background: BackgroundTasks) -> RepoStatusOut:
    """Accept a GitHub URL and start indexing in the background.

    Returns 202 immediately: cloning and embedding take minutes, so the client
    polls GET /repos/{id} for the stage rather than holding a request open.
    """
    try:
        parsed = parse_github_url(request.url)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry = get_registry()
    collection = collection_name_for_repo(parsed.slug, request.variant)

    existing = registry.get_by_collection(collection)
    if existing and existing.status in {"queued", "cloning", "indexing"}:
        # Idempotent: re-submitting a repo already being indexed is not an error.
        return to_status(existing)
    if existing:
        registry.update_repo(existing.id, status=QUEUED, stage="queued", error=None)
        repo = registry.get_repo(existing.id)
    else:
        repo = registry.create_repo(
            name=parsed.slug,
            variant=request.variant,
            collection=collection,
            url=request.url,
            status=QUEUED,
        )

    background.add_task(run_ingest, repo.id, request.url)
    return to_status(repo)


@router.get("/{repo_id}", response_model=RepoStatusOut)
def get_repo(repo_id: str) -> RepoStatusOut:
    return to_status(require_repo(repo_id))


@router.post("/{repo_id}/reindex", response_model=RepoStatusOut, status_code=202)
def reindex(repo_id: str, background: BackgroundTasks) -> RepoStatusOut:
    """Re-index in place. Re-clones when the repo came from a URL, so a stale
    working tree is refreshed rather than re-embedded as-is."""
    repo = require_repo(repo_id)
    registry = get_registry()
    registry.update_repo(repo_id, status=QUEUED, stage="queued", error=None)
    background.add_task(run_ingest, repo_id, repo.url)
    return to_status(registry.get_repo(repo_id))


@router.delete("/{repo_id}", status_code=204)
def delete_repo(repo_id: str, remove_clone: bool = True) -> None:
    """Drop the indexes and registry row.

    The working tree is only removed when this tool cloned it — a repo indexed
    from a local path must never be deleted from under the user.
    """
    require_repo(repo_id)
    delete_repo_data(repo_id, remove_clone=remove_clone)
