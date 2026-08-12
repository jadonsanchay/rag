"""Hand-rolled in-memory rate limiting — no new dependency (`slowapi` was the
alternative, skipped to keep this whole plan's zero-new-Python-deps property).

Only matters once there is a public URL in front of an OpenAI-backed endpoint —
scoped to the two controls that actually bound the cost risk: a per-IP window
and a global ceiling. Both reset on process restart, which is an accepted
tradeoff for a single-instance demo (see the plan/README), not a production
guarantee. The real backstop is a hard spend cap set in the OpenAI dashboard.
"""

import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

PER_IP_LIMIT = int(os.environ.get("RATE_LIMIT_PER_IP", "20"))
PER_IP_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
GLOBAL_REQUEST_CEILING = int(os.environ.get("GLOBAL_REQUEST_CEILING", "2000"))


def is_limited_path(method: str, path: str) -> bool:
    """Only the LLM/embedding-backed endpoints are limited — /health, /metrics,
    and read-only GETs stay free."""
    if method != "POST":
        return False
    if path == "/api/ask":
        return True
    if path.startswith("/api/conversations/") and path.endswith("/messages"):
        return True
    if path == "/api/repos":
        return True
    return False


class RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._global_total = 0

    def check(self, client_ip: str) -> Optional[str]:
        """None if the request is allowed, else a client-facing rejection message."""
        now = time.time()
        with self._lock:
            self._global_total += 1
            if self._global_total > GLOBAL_REQUEST_CEILING:
                return "This demo has hit its request ceiling for now. Please try again later."

            hits = self._hits[client_ip]
            while hits and now - hits[0] > PER_IP_WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= PER_IP_LIMIT:
                return f"Rate limit exceeded: max {PER_IP_LIMIT} requests per {PER_IP_WINDOW_SECONDS}s per IP."
            hits.append(now)
            return None


limiter = RateLimiter()


def client_ip(request) -> str:
    forwarded = request.headers.get("fly-client-ip") or request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
