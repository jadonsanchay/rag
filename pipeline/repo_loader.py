from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from . import config

BINARY_SNIFF_BYTES = 8192


@dataclass
class WalkStats:
    """Why files were kept or dropped. Ignore rules drive retrieval quality,
    so they should be observable rather than silent."""

    loaded: int = 0
    skipped: Counter = field(default_factory=Counter)

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())

    def report(self) -> str:
        lines = [f"Loaded {self.loaded} files, skipped {self.total_skipped}"]
        for reason, count in self.skipped.most_common():
            lines.append(f"  skipped ({reason}): {count}")
        return "\n".join(lines)


def detect_language(path: Path) -> str:
    return config.LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "unknown")


def _is_ignored_dir(name: str) -> bool:
    return name in config.IGNORE_DIRS or name.endswith(".egg-info")


def _matches_prefix(rel_path: Path, prefixes: Sequence[str]) -> bool:
    """True if rel_path sits under any of the given repo-relative prefixes,
    so `docs/en` matches `docs/en/**` but not `docs/es/**`."""
    parts = rel_path.parts
    for prefix in prefixes:
        prefix_parts = Path(prefix).parts
        if parts[: len(prefix_parts)] == prefix_parts:
            return True
    return False


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(BINARY_SNIFF_BYTES)
    except OSError:
        return True


def _skip_reason(path: Path) -> Optional[str]:
    name = path.name
    if name in config.IGNORE_FILENAMES:
        return "ignored filename"
    if any(name.endswith(suffix) for suffix in config.IGNORE_FILE_SUFFIXES):
        return "ignored extension"
    if detect_language(path) == "unknown":
        return "unknown extension"

    try:
        size = path.stat().st_size
    except OSError:
        return "stat failed"

    if size == 0:
        return "empty file"
    if size > config.MAX_FILE_BYTES:
        return "too large"
    if _looks_binary(path):
        return "binary"
    return None


def iter_repo_files(
    repo_path: Path,
    extra_excludes: Sequence[str] = (),
    include_only: Sequence[str] = (),
) -> Iterator[Tuple[Path, Optional[str]]]:
    """Yield (path, skip_reason) for every candidate file under repo_path.
    skip_reason is None for files that should be indexed.

    include_only, when non-empty, restricts indexing to those subtrees. That
    makes an eval corpus reproducible instead of "whatever was on disk"."""
    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue

        rel_path = path.relative_to(repo_path)
        if any(_is_ignored_dir(part) for part in rel_path.parts[:-1]):
            continue
        if include_only and not _matches_prefix(rel_path, include_only):
            yield path, "outside include list"
            continue
        if _matches_prefix(rel_path, extra_excludes):
            yield path, "user excluded"
            continue

        yield path, _skip_reason(path)


def load_repo_documents(
    repo_path: Path,
    extra_excludes: Sequence[str] = (),
    include_only: Sequence[str] = (),
) -> Tuple[List[Document], WalkStats]:
    """Load an entire repo as one Document per file, with metadata that
    later stages (chunking, citations) depend on."""
    repo_path = repo_path.resolve()
    if not repo_path.is_dir():
        raise ValueError(f"Not a directory: {repo_path}")

    documents: List[Document] = []
    stats = WalkStats()

    for path, skip_reason in iter_repo_files(repo_path, extra_excludes, include_only):
        if skip_reason:
            stats.skipped[skip_reason] += 1
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            stats.skipped["decode error"] += 1
            continue

        rel_path = path.relative_to(repo_path)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "path": str(rel_path),
                    "abs_path": str(path),
                    "language": detect_language(path),
                    "ext": path.suffix.lower(),
                    "size_bytes": len(content.encode("utf-8")),
                    "repo": repo_path.name,
                },
            )
        )
        stats.loaded += 1

    return documents, stats


def language_stats(documents: Sequence[Document]) -> Counter:
    return Counter(doc.metadata.get("language", "unknown") for doc in documents)
