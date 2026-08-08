"""Access to the index manifests written by index_repo.py.

Three places needed to know where an indexed repo actually lives on disk, and all
three were assuming `REPOS_DIR/<repo>`. That assumption breaks for any repo indexed
from elsewhere — citation verification silently reported "file not found" for every
chunk. The manifest already records the real path, so it is the single source of
truth here.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .vector_store import collection_name_for_repo

MANIFEST_DIR = config.DATA_DIR / "index_manifests"


def manifest_path(collection: str) -> Path:
    return MANIFEST_DIR / f"{collection}.json"


def load_manifest(collection: str) -> Dict[str, Any]:
    path = manifest_path(collection)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_manifest_for(repo: str, variant: str) -> Dict[str, Any]:
    return load_manifest(collection_name_for_repo(repo, variant))


def all_manifests() -> List[Dict[str, Any]]:
    return [
        manifest
        for path in sorted(MANIFEST_DIR.glob("*.json"))
        if (manifest := load_manifest(path.stem))
    ]


def repo_root_for(repo: str, variant: Optional[str] = None) -> Optional[Path]:
    """Where the indexed working tree lives, or None if it cannot be found.

    Prefers the path recorded at index time; falls back to the conventional
    location so a manifest written before this existed still resolves.
    """
    if variant:
        recorded = load_manifest_for(repo, variant).get("repo_path")
        if recorded and Path(recorded).is_dir():
            return Path(recorded)

    # Any variant of this repo will do — they all point at the same tree.
    for manifest in all_manifests():
        if manifest.get("repo") == repo:
            recorded = manifest.get("repo_path")
            if recorded and Path(recorded).is_dir():
                return Path(recorded)

    fallback = config.REPOS_DIR / repo
    return fallback if fallback.is_dir() else None
