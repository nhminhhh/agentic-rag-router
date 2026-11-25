import pytest
from rag_router.telemetry.metrics import TelemetryTracker


def test_telemetry_recording_and_percentiles():
    tracker = TelemetryTracker()

    tracker.record_query(
        intent="hybrid",
        strategy="hybrid_rrf",
        latency_ms=1.2,
        is_grounded=True,
        chunks_count=3,
    )
    tracker.record_query(
        intent="exact_lookup",
        strategy="sparse_bm25",
        latency_ms=0.8,
        is_grounded=True,
        chunks_count=1,
    )
    tracker.record_query(
        intent="no_retrieval",
        strategy="none",
        latency_ms=0.1,
        is_grounded=True,
        chunks_count=0,
    )

    summary = tracker.get_summary()

    assert summary["total_queries"] == 3
    assert summary["grounding_pass_rate"] == 1.0
    assert summary["intent_distribution"]["hybrid"] == 1
    assert summary["intent_distribution"]["exact_lookup"] == 1
    assert summary["latency_p50_ms"] > 0.0
    assert summary["avg_chunks_retrieved"] == round((3 + 1 + 0) / 3, 2)


def test_telemetry_empty_state():
    tracker = TelemetryTracker()
    summary = tracker.get_summary()

    assert summary["total_queries"] == 0
    assert summary["latency_p50_ms"] == 0.0
    assert summary["intent_distribution"] == {}
