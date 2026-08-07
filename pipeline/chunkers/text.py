import re
from typing import Callable, List, Optional

from .base import Chunk

MARKDOWN_HEADING = re.compile(r"^#{1,3} \S")


def default_token_counter(text: str) -> int:
    """Rough stand-in when no tokenizer is supplied: code averages a bit
    under 4 characters per token."""
    return max(1, len(text) // 4)


class TextChunker:
    """Line-window chunker used for non-code files and unsupported languages.

    Windows on line boundaries rather than characters so every chunk carries a
    real start/end line for citations.
    """

    def __init__(
        self,
        language: str = "text",
        max_tokens: int = 256,
        count_tokens: Optional[Callable[[str], int]] = None,
        overlap_lines: int = 2,
    ):
        self.language = language
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens or default_token_counter
        self.overlap_lines = overlap_lines

    def chunk(self, source: str, path: str) -> List[Chunk]:
        lines = source.splitlines()
        if not lines:
            return []

        sections = (
            self._split_markdown_sections(lines)
            if self.language == "markdown"
            else [(1, lines)]
        )

        chunks: List[Chunk] = []
        for start_line, section_lines in sections:
            chunks.extend(self._window(section_lines, start_line, path))
        return chunks

    def _split_markdown_sections(self, lines: List[str]) -> List[tuple]:
        """Prefer heading boundaries so a chunk is a coherent doc section."""
        sections: List[tuple] = []
        current: List[str] = []
        current_start = 1

        for offset, line in enumerate(lines, start=1):
            if MARKDOWN_HEADING.match(line) and current:
                sections.append((current_start, current))
                current, current_start = [line], offset
            else:
                current.append(line)

        if current:
            sections.append((current_start, current))
        return sections

    def _window(self, lines: List[str], start_line: int, path: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        index = 0
        # Per-line costs avoid re-tokenizing the growing window each iteration.
        line_costs = [self.count_tokens(line) + 1 for line in lines]

        while index < len(lines):
            cursor = index
            running = 0

            while cursor < len(lines):
                if cursor > index and running + line_costs[cursor] > self.max_tokens:
                    break
                running += line_costs[cursor]
                cursor += 1

            text = "\n".join(lines[index:cursor]).strip()
            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        path=path,
                        start_line=start_line + index,
                        end_line=start_line + cursor - 1,
                        kind="text",
                        language=self.language,
                    )
                )

            if cursor >= len(lines):
                break
            # Always advance, even if a single line blew the budget.
            index = max(cursor - self.overlap_lines, index + 1)

        return chunks
