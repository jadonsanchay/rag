import ast
from typing import Callable, List, Optional, Tuple

from .base import Chunk
from .text import TextChunker, default_token_counter

DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPE_NODES = DEF_NODES + (ast.ClassDef,)


class PythonASTChunker:
    """Chunks Python at function/class boundaries using the stdlib ast module.

    Produces four chunk kinds:
      - function        a function or method body, decorators included
      - class_skeleton  class signature, docstring and method signatures, so
                        class-level questions retrieve without pulling every method
      - module          contiguous runs of top-level code (imports, constants)
      - text            fallback when the file does not parse

    Oversized chunks are windowed with the signature re-attached to each part,
    because the embedding model truncates anything past its input limit.
    """

    language = "python"

    def __init__(
        self,
        max_tokens: int = 256,
        count_tokens: Optional[Callable[[str], int]] = None,
    ):
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens or default_token_counter

    def chunk(self, source: str, path: str) -> List[Chunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Unparseable (e.g. py2 or templated) — still index it.
            fallback = TextChunker("python", self.max_tokens, self.count_tokens)
            return fallback.chunk(source, path)

        lines = source.splitlines()
        chunks: List[Chunk] = []

        for node in tree.body:
            if isinstance(node, DEF_NODES):
                chunks.extend(self._function_chunks(node, lines, path, parent=None))
            elif isinstance(node, ast.ClassDef):
                chunks.extend(self._class_chunks(node, lines, path))

        chunks.extend(self._module_chunks(tree, lines, path))
        return [c for c in chunks if c.text.strip()]

    # --- node handlers ------------------------------------------------------

    def _function_chunks(
        self, node: ast.AST, lines: List[str], path: str, parent: Optional[str]
    ) -> List[Chunk]:
        start, end = self._node_span(node)
        text = self._slice(lines, start, end)
        header = self._signature_header(node, lines)
        return self._emit(
            text=text,
            path=path,
            start_line=start,
            end_line=end,
            kind="function",
            symbol=node.name,
            parent_symbol=parent,
            header=header,
        )

    def _class_chunks(self, node: ast.ClassDef, lines: List[str], path: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        start, end = self._node_span(node)

        # Anchor the skeleton to the class header, not the whole class body.
        # APIRouter spans lines 593-4437; citing that range is useless, and the
        # skeleton's content is the signature block anyway.
        anchor_end = self._header_anchor_end(node, start)

        skeleton = self._emit(
            text=self._class_skeleton(node, lines),
            path=path,
            start_line=start,
            end_line=anchor_end,
            kind="class_skeleton",
            symbol=node.name,
            parent_symbol=None,
            header=f"class {node.name}",
        )
        for chunk in skeleton:
            chunk.extra["class_end_line"] = end
        chunks.extend(skeleton)

        for child in node.body:
            if isinstance(child, DEF_NODES):
                chunks.extend(self._function_chunks(child, lines, path, parent=node.name))

        return chunks

    def _module_chunks(self, tree: ast.Module, lines: List[str], path: str) -> List[Chunk]:
        """Contiguous runs of top-level non-def code, so imports and constants
        stay retrievable with an honest line range."""
        runs: List[List[ast.stmt]] = []
        current: List[ast.stmt] = []

        for node in tree.body:
            if isinstance(node, SCOPE_NODES):
                if current:
                    runs.append(current)
                    current = []
            else:
                current.append(node)
        if current:
            runs.append(current)

        chunks: List[Chunk] = []
        for run in runs:
            start = min(self._node_span(n)[0] for n in run)
            end = max(self._node_span(n)[1] for n in run)
            chunks.extend(
                self._emit(
                    text=self._slice(lines, start, end),
                    path=path,
                    start_line=start,
                    end_line=end,
                    kind="module",
                    symbol=None,
                    parent_symbol=None,
                    header=f"# {path} (module level)",
                )
            )
        return chunks

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _node_span(node: ast.AST) -> Tuple[int, int]:
        """1-indexed inclusive line span, including decorators."""
        start = node.lineno
        for decorator in getattr(node, "decorator_list", []) or []:
            start = min(start, decorator.lineno)
        return start, getattr(node, "end_lineno", None) or node.lineno

    @staticmethod
    def _slice(lines: List[str], start: int, end: int) -> str:
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def _header_anchor_end(node: ast.AST, start: int) -> int:
        """Last line of the definition header: through the docstring if there is
        one, otherwise through the signature."""
        body = getattr(node, "body", None)
        if not body:
            return start
        first = body[0]
        is_docstring = isinstance(first, ast.Expr) and isinstance(
            getattr(first, "value", None), ast.Constant
        )
        if is_docstring:
            return getattr(first, "end_lineno", None) or first.lineno
        return max(start, first.lineno - 1)

    @staticmethod
    def _first_signature_line(node: ast.AST, lines: List[str]) -> str:
        """Just the `def name(` / `class Name(` line.

        Deliberately not the full signature: FastAPI's applications.py has
        signatures spanning hundreds of lines of Annotated params, and a header
        that large would dwarf the content it is supposed to label.
        """
        line = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
        return line if line.endswith((":", "(")) else line

    def _signature_header(self, node: ast.AST, lines: List[str]) -> str:
        """Compact identifier for a windowed part: the def line plus the first
        docstring line."""
        header = self._first_signature_line(node, lines)
        docstring = ast.get_docstring(node) if isinstance(node, SCOPE_NODES) else None
        if docstring:
            first_line = docstring.strip().splitlines()[0][:160]
            header = f'{header}\n    """{first_line}"""'
        return header

    def _class_skeleton(self, node: ast.ClassDef, lines: List[str]) -> str:
        """Class signature + docstring + method signature lines, so class-level
        questions retrieve without dragging in every method body."""
        parts = [self._signature_header(node, lines)]
        for child in node.body:
            if isinstance(child, DEF_NODES):
                parts.append(f"    {self._first_signature_line(child, lines)}")
        return "\n".join(parts)

    def _emit(
        self,
        text: str,
        path: str,
        start_line: int,
        end_line: int,
        kind: str,
        symbol: Optional[str],
        parent_symbol: Optional[str],
        header: str,
    ) -> List[Chunk]:
        """One chunk if it fits the token budget, otherwise windowed parts
        that each re-state the signature."""
        if self.count_tokens(text) <= self.max_tokens:
            return [
                Chunk(
                    text=text,
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    kind=kind,
                    language=self.language,
                    symbol=symbol,
                    parent_symbol=parent_symbol,
                )
            ]

        lines = text.splitlines()
        # Bound the header so it can never dominate the part it labels.
        header = header[:300]
        header_cost = min(self.count_tokens(header), self.max_tokens // 4)
        budget = max(self.max_tokens - header_cost, self.max_tokens // 4)

        # Per-line costs, summed. Re-tokenizing the whole window on every line
        # is quadratic; summing per-line counts slightly over-estimates, which
        # errs on the safe side of the limit.
        line_costs = [self.count_tokens(line) + 1 for line in lines]

        windows: List[Tuple[int, int, str]] = []
        index = 0
        while index < len(lines):
            cursor = index
            running = 0
            while cursor < len(lines):
                if cursor > index and running + line_costs[cursor] > budget:
                    break
                running += line_costs[cursor]
                cursor += 1
            windows.append((index, cursor, "\n".join(lines[index:cursor])))
            if cursor >= len(lines):
                break
            index = cursor

        total = len(windows)
        header_lines = len(header.splitlines())
        chunks: List[Chunk] = []
        for part_number, (offset, cursor, body) in enumerate(windows, start=1):
            is_continuation = part_number > 1
            prefix = f"{header}\n" if is_continuation else ""
            chunks.append(
                Chunk(
                    text=f"{prefix}{body}".strip(),
                    path=path,
                    start_line=start_line + offset,
                    end_line=start_line + cursor - 1,
                    kind=kind,
                    language=self.language,
                    symbol=symbol,
                    parent_symbol=parent_symbol,
                    part=part_number,
                    part_count=total,
                    # Continuation parts get a synthetic identifier header that
                    # is NOT present at start_line. A source viewer must skip
                    # these lines when mapping chunk text back to the file.
                    extra={"header_lines": header_lines if is_continuation else 0},
                )
            )
        return chunks
