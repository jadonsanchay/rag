import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Overridable so a single mounted volume (e.g. a Fly.io volume) can cover the
# vector store, lexical index, registry, and cloned repos together in one place.
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", BASE_DIR / "data"))
PDF_DIR = DATA_DIR / "pdf_files"
TEXT_DIR = DATA_DIR / "text_files"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LEXICAL_INDEX_DIR = DATA_DIR / "lexical_index"
# Cloned repos live under DATA_DIR (not the project tree): a vendored
# pyproject.toml inside a clone would make uv treat it as a local package
# source and rebuild the venv, so it must stay outside BASE_DIR either way.
REPOS_DIR = DATA_DIR / "repos"
EVAL_DIR = BASE_DIR / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

COLLECTION_NAME = "pdf_documents"

# --- Embeddings -------------------------------------------------------------
# Provider is swappable so the eval harness can measure one variable at a time.
# Defaults reflect the Phase A winner (see eval/RESULTS.md): structural
# chunking for code plus a long-context embedding model.
EMBEDDING_PROVIDER = "openai"  # or "sentence-transformers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Known input limits, in tokens. Chunkers ask the provider for this rather
# than hardcoding a size, so swapping models cannot silently truncate.
MODEL_TOKEN_LIMITS = {
    "all-MiniLM-L6-v2": 256,
    "BAAI/bge-small-en-v1.5": 512,
    "jinaai/jina-embeddings-v2-base-code": 8192,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
}

# --- Chunking ---------------------------------------------------------------
CHUNK_STRATEGY = "ast-code"  # "text" reproduces the step 1 baseline
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# A model's token limit is a CEILING, not a target size. Embedding a very long
# passage averages it into a muddy vector that matches nothing specifically, so
# retrieval chunks are deliberately far smaller than the model allows.
TARGET_PROSE_TOKENS = 400  # markdown/text: no structure worth preserving
MAX_CODE_CHUNK_TOKENS = 1200  # keeps most functions whole, splits monsters

OPENAI_MODEL = "gpt-4o-mini"

# --- Retrieval --------------------------------------------------------------
# Step 4 result (eval/RESULTS.md). Lexical must outweigh semantic; equal
# weighting is actively worse than lexical alone because it dilutes the stronger
# ranking. These weights are tuned for recall@5 rather than MRR: the generator
# is handed ~6 chunks, so what matters is whether the right file is in the
# context at all, not whether it ranked first.
RETRIEVAL_MODE = "hybrid"  # semantic | lexical | hybrid
SEMANTIC_WEIGHT = 1.0
LEXICAL_WEIGHT = 2.0
RRF_K = 60
CANDIDATE_K = 40

# Step 5 result: cap chunks per file in the result set. One docs page was taking
# 4 of 5 context slots; capping it lifted architectural recall@10 from 0.600 to
# 0.800 by making room for the other files that answer the question.
MAX_CHUNKS_PER_FILE = 2

# Index structural file/package cards (step 5). Architectural questions are
# about relationships between files, which no single code chunk contains.
INDEX_CARDS = True

# Prose outnumbers code 5:1 at the chunk level (2201 vs 430 on the fastapi
# corpus), so docs crowd source files out of the top-k regardless of relevance.
# Stratified retrieval gives code and prose separate rank spaces and fuses them,
# so the best code chunk competes at rank 1 of its own list.
STRATIFY_RETRIEVAL = True
CODE_LANGUAGES = {
    "python", "javascript", "typescript", "go", "java", "ruby", "rust",
    "c", "cpp", "csharp", "php", "swift", "kotlin", "scala", "shell", "sql",
}
# Do NOT tune these away from parity. Equal weighting is the only stable point:
# any tilt makes one stratum systematically outrank the other at equal
# within-stratum rank, and since each stratum supplies more candidates than there
# are slots, the lighter stratum is shut out. prose_weight=1.1 alone drops overall
# MRR from 0.706 to 0.349.
CODE_WEIGHT = 1.0
PROSE_WEIGHT = 1.0

# --- Repo walking -----------------------------------------------------------
MAX_FILE_BYTES = 1_000_000

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "site-packages",
    "dist",
    "build",
    "target",
    "vendor",
    ".next",
    ".nuxt",
    ".idea",
    ".vscode",
    "coverage",
    "htmlcov",
    ".egg-info",
}

IGNORE_FILE_SUFFIXES = {
    ".lock",
    ".map",
    ".min.js",
    ".min.css",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".a",
    ".o",
    ".class",
    ".jar",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".bin",
    ".pkl",
    ".npy",
    ".pt",
    ".onnx",
    ".safetensors",
}

IGNORE_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "Cargo.lock",
    "composer.lock",
    "Gemfile.lock",
}

# Extension -> language. Drives chunker dispatch in step 3 and the
# LanguageAdapter registry in step 9.
LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
}
