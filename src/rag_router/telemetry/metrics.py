from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List


class TelemetryTracker:
    """
    In-memory telemetry and observability tracker for Adaptive RAG Router.
    Records routing latencies, intent distributions, and grounding verification health.
    """

    def __init__(self):
        self.latencies_ms: List[float] = []
        self.intents: Counter = Counter()
        self.strategies: Counter = Counter()
        self.grounded_counts: int = 0
        self.total_queries: int = 0
        self.chunks_retrieved: List[int] = []

    def record_query(
        self,
        intent: str,
        strategy: str,
        latency_ms: float,
        is_grounded: bool,
        chunks_count: int,
    ) -> None:
        self.total_queries += 1
        self.latencies_ms.append(latency_ms)
        self.intents[intent] += 1
        self.strategies[strategy] += 1
        if is_grounded:
            self.grounded_counts += 1
        self.chunks_retrieved.append(chunks_count)

    def _percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return round(d0 + d1, 3)

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated metrics summary dictionary."""
        if self.total_queries == 0:
            return {
                "total_queries": 0,
                "latency_p50_ms": 0.0,
                "latency_p90_ms": 0.0,
                "latency_p99_ms": 0.0,
                "grounding_pass_rate": 1.0,
                "avg_chunks_retrieved": 0.0,
                "intent_distribution": {},
                "strategy_distribution": {},
            }

        return {
            "total_queries": self.total_queries,
            "latency_p50_ms": self._percentile(self.latencies_ms, 50),
            "latency_p90_ms": self._percentile(self.latencies_ms, 90),
            "latency_p99_ms": self._percentile(self.latencies_ms, 99),
            "latency_avg_ms": round(statistics.mean(self.latencies_ms), 3),
            "grounding_pass_rate": round(self.grounded_counts / self.total_queries, 4),
            "avg_chunks_retrieved": round(statistics.mean(self.chunks_retrieved), 2),
            "intent_distribution": dict(self.intents),
            "strategy_distribution": dict(self.strategies),
        }

    def reset(self) -> None:
        """Clears all telemetry counters."""
        self.latencies_ms.clear()
        self.intents.clear()
        self.strategies.clear()
        self.grounded_counts = 0
        self.total_queries = 0
        self.chunks_retrieved.clear()
