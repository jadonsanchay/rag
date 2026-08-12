"""Clone a repository and index it, reporting progress to the registry.

Indexing a real repo takes minutes, which rules out doing it inside a request. The
job therefore runs in the background and writes its stage to the registry as it
goes, so a client can poll rather than hold a connection open.

The URL is untrusted input, so it is validated before anything touches the disk,
git is invoked without a shell, and the clone is bounded by depth, timeout and an
on-disk size check.
"""

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import config
from .registry import CLONING, FAILED, INDEXING, READY, get_registry

logger = logging.getLogger("codebase_qa.ingest")

# Only https GitHub URLs. Rejects ssh/git/file schemes, which could otherwise
# reach the local filesystem or a private host.
GITHUB_URL = re.compile(
    r"^https://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)

CLONE_TIMEOUT_SECONDS = 300
# Env-overridable so the deployed instance (fly.toml) can run tighter caps
# than local dev without a code change.
MAX_REPO_MB = int(os.environ.get("MAX_REPO_MB", "400"))
MAX_INDEXED_FILES = int(os.environ.get("MAX_INDEXED_FILES", "6000"))


class IngestError(Exception):
    """Something went wrong that the user should see."""


@dataclass
class ParsedRepo:
    owner: str
    name: str
    url: str

    @property
    def slug(self) -> str:
        return f"{self.owner}-{self.name}"


def parse_github_url(url: str) -> ParsedRepo:
    match = GITHUB_URL.match(url.strip())
    if not match:
        raise IngestError(
            "Only public GitHub HTTPS URLs are supported, "
            "e.g. https://github.com/owner/repo"
        )
    owner, name = match.group("owner"), match.group("repo")
    # `..` in a path segment could escape the clone directory.
    if ".." in owner or ".." in name:
        raise IngestError("Invalid repository path")
    return ParsedRepo(owner=owner, name=name, url=f"https://github.com/{owner}/{name}.git")


def directory_size_mb(path: Path) -> float:
    total = 0
    for file in path.rglob("*"):
        if file.is_file() and not file.is_symlink():
            try:
                total += file.stat().st_size
            except OSError:
                continue
    return total / (1024 * 1024)


def clone_repo(parsed: ParsedRepo, destination: Path) -> str:
    """Shallow-clone and return the commit SHA."""
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--single-branch",
                "--no-tags",
                parsed.url,
                str(destination),
            ],
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
            # No shell: the URL is user input.
            env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(destination, ignore_errors=True)
        raise IngestError(f"Clone timed out after {CLONE_TIMEOUT_SECONDS}s")

    if result.returncode != 0:
        shutil.rmtree(destination, ignore_errors=True)
        detail = (result.stderr or "").strip().splitlines()
        message = detail[-1] if detail else "git clone failed"
        raise IngestError(f"Could not clone: {message[:200]}")

    size_mb = directory_size_mb(destination)
    if size_mb > MAX_REPO_MB:
        shutil.rmtree(destination, ignore_errors=True)
        raise IngestError(
            f"Repository is {size_mb:.0f}MB, over the {MAX_REPO_MB}MB limit"
        )

    sha = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    )
    return sha.stdout.strip() if sha.returncode == 0 else "unknown"


def index_repo_path(
    repo_path: Path,
    collection: str,
    on_stage: Optional[Callable[[str], None]] = None,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> dict:
    """Run the indexing pipeline for an already-present working tree.

    Kept separate from cloning so a local path or a re-index can reuse it.
    """
    from .cards import build_cards
    from .chunkers import chunk_documents
    from .embeddings import EmbeddingManager
    from .ids import chunk_ids_for
    from .lexical_index import LexicalIndex
    from .repo_loader import language_stats, load_repo_documents
    from .splitter import split_documents
    from .vector_store import VectorStore

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    stage("loading files")
    documents, walk_stats = load_repo_documents(repo_path, exclude, include)
    if not documents:
        raise IngestError("No indexable files found in this repository")
    if len(documents) > MAX_INDEXED_FILES:
        raise IngestError(
            f"{len(documents)} indexable files, over the {MAX_INDEXED_FILES} limit. "
            "Index a subdirectory instead."
        )

    stage("chunking")
    embedder = EmbeddingManager()
    code_docs = [
        d for d in documents if d.metadata.get("language") in config.CODE_LANGUAGES
    ]
    prose_docs = [
        d for d in documents if d.metadata.get("language") not in config.CODE_LANGUAGES
    ]
    chunks = split_documents(prose_docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    chunks += chunk_documents(
        code_docs,
        token_limit=embedder.token_limit,
        count_tokens=embedder.count_tokens,
    )
    if config.INDEX_CARDS:
        chunks += build_cards(
            documents,
            max_tokens=config.TARGET_PROSE_TOKENS,
            count_tokens=embedder.count_tokens,
            file_chunks=chunks,
        )
    if not chunks:
        raise IngestError("Files were found but produced no chunks")

    stage(f"embedding {len(chunks)} chunks")
    embeddings = embedder.generate_embeddings([c.page_content for c in chunks])

    stage("writing indexes")
    store = VectorStore(collection_name=collection)
    store.reset()
    ids = chunk_ids_for(chunks)
    store.add_documents(chunks, embeddings, ids=ids)

    lexical = LexicalIndex(collection)
    lexical.reset()
    lexical.add_documents(chunks, ids)

    return {
        "files_indexed": len(documents),
        "chunks": len(chunks),
        "languages": dict(language_stats(documents)),
        "skipped": dict(walk_stats.skipped),
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "embedding_model": embedder.model_name,
    }


def run_ingest(repo_id: str, url: Optional[str] = None) -> None:
    """Background entrypoint: clone if needed, then index, updating status.

    Every failure path must land in the registry as `failed` with a message — a
    job that dies silently looks identical to one still running.
    """
    from .vector_store import collection_name_for_repo

    registry = get_registry()
    repo = registry.get_repo(repo_id)
    if repo is None:
        return

    started = time.time()
    try:
        repo_path = Path(repo.repo_path) if repo.repo_path else None

        if url:
            parsed = parse_github_url(url)
            repo_path = config.REPOS_DIR / parsed.slug
            registry.update_repo(repo_id, status=CLONING, stage="cloning", error=None)
            commit_sha = clone_repo(parsed, repo_path)
            registry.update_repo(
                repo_id, repo_path=str(repo_path), commit_sha=commit_sha
            )
        elif repo_path is None or not repo_path.is_dir():
            raise IngestError("Repository is no longer on disk; re-add it by URL")

        registry.update_repo(repo_id, status=INDEXING, stage="starting")
        stats = index_repo_path(
            repo_path,
            repo.collection or collection_name_for_repo(repo.name, repo.variant),
            on_stage=lambda name: registry.update_repo(repo_id, stage=name),
        )

        # By collection, not by id: multiple users can have their own row over
        # this same shared index, and all of them need the refreshed stats,
        # not just the row that happened to trigger this job.
        registry.update_repos_by_collection(
            repo.collection,
            status=READY,
            stage=None,
            error=None,
            indexed_at=time.time(),
            files_indexed=stats["files_indexed"],
            chunks=stats["chunks"],
            languages=stats["languages"],
            embedding_provider=stats["embedding_provider"],
            embedding_model=stats["embedding_model"],
            size_bytes=int(directory_size_mb(repo_path) * 1024 * 1024),
        )
        logger.info(
            "ingest ready",
            extra={
                "repo_id": repo_id,
                "repo": repo.name,
                "duration_s": round(time.time() - started, 1),
                "chunks": stats["chunks"],
            },
        )
    except IngestError as exc:
        registry.update_repos_by_collection(
            repo.collection, status=FAILED, stage=None, error=str(exc)
        )
        logger.error(
            "ingest failed", extra={"repo_id": repo_id, "repo": repo.name, "reason": str(exc)}
        )
    except Exception as exc:  # noqa: BLE001 - never leave a job in limbo
        registry.update_repos_by_collection(
            repo.collection, status=FAILED, stage=None, error=f"{type(exc).__name__}: {exc}"
        )
        logger.exception("ingest crashed", extra={"repo_id": repo_id, "repo": repo.name})


def delete_repo_data(
    repo_id: str, remove_clone: bool = True, teardown_index: bool = True
) -> None:
    """Drop a repo's registry row, and — only when `teardown_index` — its
    indexes and (optionally) its working tree too.

    `teardown_index=False` when another user's row still points at the same
    collection: that user's repo list depends on this Chroma collection,
    lexical index, and clone still existing, so only *this* row goes away.
    """
    from .lexical_index import LexicalIndex
    from .vector_store import VectorStore

    registry = get_registry()
    repo = registry.get_repo(repo_id)
    if repo is None:
        return

    if teardown_index:
        try:
            VectorStore(collection_name=repo.collection).client.delete_collection(
                repo.collection
            )
        except Exception:  # noqa: BLE001 - collection may not exist
            pass

        lexical_db = LexicalIndex(repo.collection).db_path
        try:
            lexical_db.unlink(missing_ok=True)
        except OSError:
            pass

        # Only remove clones this tool created; never a user's own working tree.
        if remove_clone and repo.repo_path:
            path = Path(repo.repo_path)
            if path.is_dir() and config.REPOS_DIR in path.parents:
                shutil.rmtree(path, ignore_errors=True)

    registry.delete_repo(repo_id)
