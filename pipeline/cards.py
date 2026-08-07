"""File and package cards: chunks that describe structure rather than contain code.

Architectural questions ("how is the security system organized?", "how does a
request flow from ASGI entry to the endpoint?") have no single chunk that answers
them — the answer is a *relationship between* files. Function-level chunks can
never surface it, which is why architectural retrieval stayed the weakest segment
through steps 1-4.

Cards are built heuristically from the AST, not by an LLM: summarising every file
with a model would cost an API call per file and add minutes to indexing, for
information that imports and symbol names already carry.
"""

import ast
import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from .chunkers.base import Chunk
from .chunkers.text import default_token_counter

MARKDOWN_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

MAX_SYMBOLS_LISTED = 25
MAX_IMPORTS_LISTED = 20
MAX_DOCSTRING_CHARS = 300
MAX_FILES_PER_PACKAGE_CARD = 18


def _dedupe(items: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(item for item in items if item))


def _first_sentence(text: str, limit: int = MAX_DOCSTRING_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:limit]


class PythonStructure:
    """Top-level structure of a Python module, extracted with the stdlib ast."""

    def __init__(self, source: str):
        self.docstring = ""
        self.imports: List[str] = []
        self.classes: Dict[str, List[str]] = {}
        self.functions: List[str] = []
        self.constants: List[str] = []
        self.parsed = False

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return

        self.parsed = True
        self.docstring = _first_sentence(ast.get_docstring(tree) or "")

        for node in tree.body:
            if isinstance(node, ast.Import):
                self.imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imports.append(node.module)
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = [
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                ]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.append(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        self.constants.append(target.id)

    @property
    def symbols(self) -> List[str]:
        return list(self.classes) + self.functions


def python_file_card(source: str, path: str) -> Optional[str]:
    """A prose-ish structural summary, phrased so both BM25 and embeddings can
    match it: the path, what it defines, and what it depends on."""
    structure = PythonStructure(source)
    if not structure.parsed:
        return None
    if not (structure.symbols or structure.docstring or structure.imports):
        return None

    posix = PurePosixPath(path)
    package = str(posix.parent) if str(posix.parent) != "." else "(root)"
    lines = [f"{path} — Python module in package {package}."]

    if structure.docstring:
        lines.append(f"Purpose: {structure.docstring}")

    if structure.classes:
        described = []
        for name, methods in list(structure.classes.items())[:MAX_SYMBOLS_LISTED]:
            if methods:
                shown = ", ".join(methods[:8])
                described.append(f"{name} (methods: {shown})")
            else:
                described.append(name)
        lines.append("Defines classes: " + "; ".join(described) + ".")

    if structure.functions:
        shown = ", ".join(structure.functions[:MAX_SYMBOLS_LISTED])
        lines.append(f"Defines functions: {shown}.")

    if structure.constants:
        shown = ", ".join(_dedupe(structure.constants)[:MAX_SYMBOLS_LISTED])
        lines.append(f"Defines constants: {shown}.")

    if structure.imports:
        shown = ", ".join(_dedupe(structure.imports)[:MAX_IMPORTS_LISTED])
        lines.append(f"Depends on: {shown}.")

    return "\n".join(lines)


def markdown_file_card(source: str, path: str) -> Optional[str]:
    """Heading outline for a docs page, so 'how do I do X' can match a page's
    overall shape rather than one paragraph of it."""
    headings = [text.strip() for _, text in MARKDOWN_HEADING.findall(source)]
    if not headings:
        return None
    shown = "; ".join(_dedupe(headings)[:MAX_SYMBOLS_LISTED])
    return f"{path} — documentation page.\nSections: {shown}."


def build_file_cards(documents: Sequence[Document]) -> List[Chunk]:
    cards: List[Chunk] = []

    for document in documents:
        metadata = document.metadata
        path = metadata.get("path", "unknown")
        language = metadata.get("language", "text")

        if language == "python":
            text = python_file_card(document.page_content, path)
        elif language == "markdown":
            text = markdown_file_card(document.page_content, path)
        else:
            text = None

        if not text:
            continue

        cards.append(
            Chunk(
                text=text,
                path=path,
                start_line=1,
                end_line=len(document.page_content.splitlines()) or 1,
                kind="file_card",
                language=language,
                extra={"is_card": True},
            )
        )

    return cards


def build_package_cards(
    documents: Sequence[Document],
    max_tokens: int = 400,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> List[Chunk]:
    """Directory-level cards.

    "How is the security system organized?" is a question about a *package*, not
    a file. One card per directory listing its modules and their main symbols
    gives that question something to match.
    """
    counter = count_tokens or default_token_counter
    by_package: Dict[str, List[tuple]] = defaultdict(list)

    for document in documents:
        if document.metadata.get("language") != "python":
            continue
        path = document.metadata.get("path", "")
        structure = PythonStructure(document.page_content)
        if not structure.parsed:
            continue
        package = str(PurePosixPath(path).parent)
        by_package[package].append((PurePosixPath(path).name, structure))

    cards: List[Chunk] = []
    for package, modules in sorted(by_package.items()):
        if len(modules) < 2:
            continue  # a one-module package adds nothing over its file card

        modules.sort()
        entries = []
        for name, structure in modules:
            symbols = structure.symbols[:10]
            if symbols:
                entries.append(f"{name} defines {', '.join(symbols)}.")
            elif structure.docstring:
                entries.append(f"{name}: {structure.docstring[:120]}")
            else:
                entries.append(f"{name} (re-exports only).")

        # Split oversized packages so no card is truncated at embed time.
        windows = [
            entries[i : i + MAX_FILES_PER_PACKAGE_CARD]
            for i in range(0, len(entries), MAX_FILES_PER_PACKAGE_CARD)
        ]
        for part_number, window in enumerate(windows, start=1):
            part_note = f" (part {part_number} of {len(windows)})" if len(windows) > 1 else ""
            header = (
                f"Package {package} — contains {len(modules)} Python modules{part_note}. "
                f"Structure and organization of {package}:"
            )
            text = header + "\n" + "\n".join(window)

            while counter(text) > max_tokens and len(window) > 1:
                window = window[:-1]
                text = header + "\n" + "\n".join(window)

            cards.append(
                Chunk(
                    text=text,
                    path=f"{package}/",
                    start_line=1,
                    end_line=1,
                    kind="package_card",
                    language="python",
                    symbol=package,
                    part=part_number if len(windows) > 1 else None,
                    part_count=len(windows) if len(windows) > 1 else None,
                    extra={"is_card": True, "module_count": len(modules)},
                )
            )

    return cards


def build_cards(
    documents: Sequence[Document],
    max_tokens: int = 400,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> List[Document]:
    """All structural cards for a corpus, as Documents ready to embed."""
    chunks = build_file_cards(documents) + build_package_cards(
        documents, max_tokens=max_tokens, count_tokens=count_tokens
    )

    by_path = {doc.metadata.get("path"): doc.metadata for doc in documents}
    result: List[Document] = []
    for chunk in chunks:
        base = {
            key: value
            for key, value in (by_path.get(chunk.path) or {}).items()
            if key not in {"path", "language", "size_bytes", "content_length"}
        }
        result.append(chunk.to_document(base))
    return result
