"""Minimal Server-Sent Events framing.

Hand-rolled rather than pulling in sse-starlette: the format is four lines of code
and the streaming contract is the interesting part, not the encoding.
"""

import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    """Encode one SSE message.

    Newlines inside `data` would terminate the frame early, so the payload is
    always JSON (which escapes them) rather than raw text.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def sse_comment(text: str = "") -> str:
    """A comment frame. Used as an early flush so proxies do not buffer the
    response before the first real event arrives."""
    return f": {text}\n\n"
