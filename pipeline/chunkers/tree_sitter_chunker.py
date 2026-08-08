"""Structural chunking for non-Python languages, via tree-sitter.

Python keeps its stdlib `ast` chunker: it understands docstrings and decorators
natively and its output is already measured. Every other language routes here.

Rather than writing a tree-sitter query file per language, each language declares
which node types are functions and which are containers (class / struct / impl /
trait / interface). That is a handful of strings per language instead of a grammar
query, and the chunking logic itself is shared — adding a language is a dict entry
plus a smoke test.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set

from .base import Chunk, emit_windowed
from .text import TextChunker, default_token_counter

# Field names tree-sitter grammars use for the declared name of a node. Rust's
# `impl_item` names its subject "type" rather than "name".
NAME_FIELDS = ("name", "type", "declarator")
NAME_NODE_TYPES = {
    "identifier",
    "type_identifier",
    "property_identifier",
    "field_identifier",
    "constant",
}


@dataclass(frozen=True)
class LanguageSpec:
    """Which node types matter, for one language."""

    functions: Set[str]
    containers: Set[str] = field(default_factory=set)
    #: Node types to descend into when looking for top-level declarations,
    #: e.g. a JS `export` statement wrapping a class.
    transparent: Set[str] = field(default_factory=set)
    #: Declaration statements whose initialiser may *be* a function, covering
    #: `const handler = () => {}`. Modern JS/TS code is written this way far more
    #: often than with `function` declarations — a real Next.js project measured
    #: here had zero `function_declaration` nodes and five `lexical_declaration`
    #: nodes per file, so omitting this makes the language support nominal.
    binding_statements: Set[str] = field(default_factory=set)
    #: Values that count as a function body when bound to a name.
    function_values: Set[str] = field(default_factory=set)


JS_BINDINGS = {"lexical_declaration", "variable_declaration"}
JS_FUNCTION_VALUES = {"arrow_function", "function", "function_expression"}


SPECS = {
    "javascript": LanguageSpec(
        functions={
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
        },
        containers={"class_declaration"},
        transparent={"export_statement", "program"},
        binding_statements=JS_BINDINGS,
        function_values=JS_FUNCTION_VALUES,
    ),
    "typescript": LanguageSpec(
        functions={
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
            "method_signature",
            "abstract_method_signature",
        },
        containers={
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "enum_declaration",
        },
        transparent={"export_statement", "program", "ambient_declaration"},
        binding_statements=JS_BINDINGS,
        function_values=JS_FUNCTION_VALUES,
    ),
    "go": LanguageSpec(
        functions={"function_declaration", "method_declaration"},
        containers={"type_declaration"},
        transparent={"source_file"},
    ),
    "rust": LanguageSpec(
        functions={"function_item", "function_signature_item"},
        containers={"impl_item", "trait_item", "struct_item", "enum_item", "mod_item"},
        transparent={"source_file"},
    ),
    "java": LanguageSpec(
        functions={"method_declaration", "constructor_declaration"},
        containers={"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
        transparent={"program"},
    ),
}

# TSX/JSX share their base grammar's node types.
SPECS["tsx"] = SPECS["typescript"]
SPECS["jsx"] = SPECS["javascript"]


def supported_languages() -> Sequence[str]:
    return sorted(SPECS)


class TreeSitterChunker:
    """Chunks a language at function and container boundaries.

    Mirrors the Python chunker's output so downstream stages need no special
    cases: `function`, `class_skeleton`, and `module` chunk kinds, methods
    carrying their container's declaration line, and oversized chunks windowed.
    """

    def __init__(
        self,
        language: str,
        max_tokens: int = 1200,
        count_tokens: Optional[Callable[[str], int]] = None,
    ):
        self.language = language
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens or default_token_counter
        self.spec = SPECS[language]
        self._parser = None

    @property
    def parser(self):
        # Parsers are built lazily and reused: constructing one per file is
        # measurable overhead across a few hundred files.
        if self._parser is None:
            from tree_sitter_language_pack import get_parser

            base = "typescript" if self.language == "tsx" else self.language
            base = "javascript" if base == "jsx" else base
            self._parser = get_parser(base)
        return self._parser

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _node_text(node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")

    @staticmethod
    def _span(node) -> tuple:
        """1-indexed inclusive line span."""
        return node.start_point[0] + 1, node.end_point[0] + 1

    @staticmethod
    def _lines_text(lines: List[str], start: int, end: int) -> str:
        """Slice by whole lines rather than by node byte offsets.

        A node's bytes can start mid-line: tree-sitter's `function_declaration`
        for `export function f()` begins at `function`, dropping `export`. Slicing
        lines keeps the chunk text byte-identical to the file at the cited range,
        which is what the citation viewer and verifier both assume.
        """
        return "\n".join(lines[start - 1 : end])

    def _name(self, node, source_bytes: bytes) -> Optional[str]:
        for field_name in NAME_FIELDS:
            child = node.child_by_field_name(field_name)
            if child is not None:
                return self._node_text(child, source_bytes).split("\n")[0][:120]
        for child in node.children:
            if child.type in NAME_NODE_TYPES:
                return self._node_text(child, source_bytes)[:120]
        # Go wraps the name a level down: `type_declaration > type_spec > name`.
        for child in node.children:
            if child.type.endswith("_spec"):
                nested = self._name(child, source_bytes)
                if nested:
                    return nested
        return None

    def _first_line(self, node, lines: List[str]) -> str:
        start = node.start_point[0]
        return lines[start].strip() if start < len(lines) else ""

    def _top_level_nodes(self, root) -> List:
        """Flatten wrappers like `export class Foo` so the class is seen."""
        out = []
        for child in root.children:
            if child.type in self.spec.transparent:
                out.extend(self._top_level_nodes(child))
            else:
                out.append(child)
        return out

    def _members(self, node) -> List:
        """Function-like descendants of a container, one level of body deep."""
        members = []
        for child in node.children:
            if child.type in self.spec.functions:
                members.append(child)
            # Class bodies / declaration lists wrap the members.
            elif child.type.endswith(("_body", "declaration_list")):
                members.extend(c for c in child.children if c.type in self.spec.functions)
        return members

    # --- chunking ----------------------------------------------------------

    def chunk(self, source: str, path: str) -> List[Chunk]:
        source_bytes = source.encode("utf-8")
        try:
            tree = self.parser.parse(source_bytes)
        except Exception:  # noqa: BLE001 - a parser failure must not lose the file
            return self._fallback(source, path)

        lines = source.splitlines()
        chunks: List[Chunk] = []
        consumed: List[tuple] = []  # line spans covered by structural chunks

        for node in self._top_level_nodes(tree.root_node):
            if node.type in self.spec.functions:
                chunks.extend(self._function_chunk(node, source_bytes, lines, path))
                consumed.append(self._span(node))
            elif node.type in self.spec.binding_statements:
                bound = self._bound_function(node, source_bytes, lines, path)
                if bound:
                    chunks.extend(bound)
                    consumed.append(self._span(node))
            elif node.type in self.spec.containers:
                chunks.extend(self._container_chunks(node, source_bytes, lines, path))
                consumed.append(self._span(node))

        chunks.extend(self._module_chunks(lines, path, consumed))
        kept = [c for c in chunks if c.text.strip()]
        # A file with no recognised declarations (config, plain script) still
        # needs to be searchable.
        return kept or self._fallback(source, path)

    def _fallback(self, source: str, path: str) -> List[Chunk]:
        return TextChunker(self.language, self.max_tokens, self.count_tokens).chunk(
            source, path
        )

    def _function_chunk(
        self,
        node,
        source_bytes: bytes,
        lines: List[str],
        path: str,
        parent: Optional[str] = None,
        container_line: Optional[str] = None,
    ) -> List[Chunk]:
        start, end = self._span(node)
        text = self._lines_text(lines, start, end)
        name = self._name(node, source_bytes)

        # Same reason as the Python chunker: without the container's declaration
        # line, a method chunk hides what it belongs to and what that extends.
        header_lines = 0
        if container_line:
            text = f"{container_line}\n{text}"
            header_lines = 1

        return emit_windowed(
            text=text,
            path=path,
            start_line=start,
            end_line=end,
            kind="function",
            language=self.language,
            header=self._first_line(node, lines),
            max_tokens=self.max_tokens,
            count_tokens=self.count_tokens,
            symbol=name,
            parent_symbol=parent,
            extra={"header_lines": header_lines} if header_lines else None,
        )

    def _bound_function(
        self, node, source_bytes: bytes, lines: List[str], path: str
    ) -> Optional[List[Chunk]]:
        """Handle `const name = () => {...}` as a named function.

        The whole statement is the chunk (so `const` and the name are included),
        but it only qualifies when the bound value is actually function-like —
        a plain `const PORT = 3000` stays module-level code.
        """
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            value = child.child_by_field_name("value")
            if value is None or value.type not in self.spec.function_values:
                continue

            name_node = child.child_by_field_name("name")
            name = self._node_text(name_node, source_bytes) if name_node else None
            start, end = self._span(node)
            return emit_windowed(
                text=self._lines_text(lines, start, end),
                path=path,
                start_line=start,
                end_line=end,
                kind="function",
                language=self.language,
                header=self._first_line(node, lines),
                max_tokens=self.max_tokens,
                count_tokens=self.count_tokens,
                symbol=name,
            )
        return None

    def _container_chunks(
        self, node, source_bytes: bytes, lines: List[str], path: str
    ) -> List[Chunk]:
        start, end = self._span(node)
        name = self._name(node, source_bytes)
        declaration = self._first_line(node, lines)
        members = self._members(node)

        skeleton_lines = [declaration]
        for member in members:
            skeleton_lines.append(f"    {self._first_line(member, lines)}")

        # Anchor the skeleton to the declaration line, not the whole body: citing
        # a 400-line span is useless to a reader.
        chunks = emit_windowed(
            text="\n".join(skeleton_lines),
            path=path,
            start_line=start,
            end_line=start,
            kind="class_skeleton",
            language=self.language,
            header=declaration,
            max_tokens=self.max_tokens,
            count_tokens=self.count_tokens,
            symbol=name,
            extra={"container_end_line": end},
        )

        for member in members:
            chunks.extend(
                self._function_chunk(
                    member, source_bytes, lines, path, parent=name,
                    container_line=declaration,
                )
            )
        return chunks

    def _module_chunks(
        self, lines: List[str], path: str, consumed: Sequence[tuple]
    ) -> List[Chunk]:
        """Contiguous runs of top-level code not already inside a declaration —
        imports, constants, top-level statements."""
        covered = set()
        for start, end in consumed:
            covered.update(range(start, end + 1))

        chunks: List[Chunk] = []
        run_start: Optional[int] = None
        run_end: Optional[int] = None

        # A blank line must not split a run, or an import block separated by blank
        # lines becomes several one-line chunks. Only a declaration boundary (or
        # end of file) closes a run; trailing blanks are trimmed via run_end.
        for number in range(1, len(lines) + 1):
            if number in covered:
                if run_start is not None:
                    chunks.extend(self._emit_module(lines, path, run_start, run_end))
                    run_start = run_end = None
                continue
            if lines[number - 1].strip():
                if run_start is None:
                    run_start = number
                run_end = number

        if run_start is not None:
            chunks.extend(self._emit_module(lines, path, run_start, run_end))
        return chunks

    def _emit_module(
        self, lines: List[str], path: str, start: int, end: int
    ) -> List[Chunk]:
        text = "\n".join(lines[start - 1 : end]).strip()
        if not text:
            return []
        return emit_windowed(
            text=text,
            path=path,
            start_line=start,
            end_line=end,
            kind="module",
            language=self.language,
            header=f"// {path} (module level)",
            max_tokens=self.max_tokens,
            count_tokens=self.count_tokens,
        )
