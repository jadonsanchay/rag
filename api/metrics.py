"""In-process counters for the `/metrics` endpoint.

Deliberately not Prometheus: there is nothing scraping this today, and standing
one up would be theater for a single-instance project. Plain JSON, reset on
restart — enough to answer "is this healthy" from the outside.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Metrics:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    requests_total: int = 0
    requests_by_status: Dict[str, int] = field(default_factory=dict)
    requests_by_path: Dict[str, int] = field(default_factory=dict)
    request_ms_sum: float = 0.0
    answers_total: int = 0
    refusals_total: int = 0
    retrieval_ms_sum: float = 0.0
    generation_ms_sum: float = 0.0
    citation_invalid_total: int = 0

    def record_request(self, path: str, status: int, duration_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_by_status[str(status)] = self.requests_by_status.get(str(status), 0) + 1
            self.requests_by_path[path] = self.requests_by_path.get(path, 0) + 1
            self.request_ms_sum += duration_ms

    def record_answer(
        self, refused: bool, retrieval_ms: int, generation_ms: int, invalid_citations: int
    ) -> None:
        with self._lock:
            self.answers_total += 1
            if refused:
                self.refusals_total += 1
            self.retrieval_ms_sum += retrieval_ms
            self.generation_ms_sum += generation_ms
            self.citation_invalid_total += invalid_citations

    def snapshot(self) -> dict:
        with self._lock:
            answers = self.answers_total or 1
            requests = self.requests_total or 1
            return {
                "requests_total": self.requests_total,
                "requests_by_status": dict(self.requests_by_status),
                "requests_by_path": dict(self.requests_by_path),
                "mean_request_ms": round(self.request_ms_sum / requests, 1),
                "answers_total": self.answers_total,
                "refusals_total": self.refusals_total,
                "mean_retrieval_ms": round(self.retrieval_ms_sum / answers, 1),
                "mean_generation_ms": round(self.generation_ms_sum / answers, 1),
                "citation_invalid_total": self.citation_invalid_total,
            }


metrics = Metrics()
