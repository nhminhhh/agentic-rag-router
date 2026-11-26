

def test_daily_routing_scenario_2026_11_25_d8376a():
    from rag_router.core.types import Chunk
    from rag_router.engine import AdaptiveRAGRouter
    router = AdaptiveRAGRouter()
    router.index([Chunk(id="c_d8376a", content="Node cluster d8376a active for 2026-11-25")])
    res = router.route_and_execute("Query regarding Node cluster d8376a")
    assert res.decision.confidence > 0.5
    assert len(res.retrieved_chunks) >= 1


def test_daily_routing_scenario_2026_11_26_1b4b80():
    from rag_router.core.types import Chunk
    from rag_router.engine import AdaptiveRAGRouter
    router = AdaptiveRAGRouter()
    router.index([Chunk(id="c_1b4b80", content="Node cluster 1b4b80 active for 2026-11-26")])
    res = router.route_and_execute("Query regarding Node cluster 1b4b80")
    assert res.decision.confidence > 0.5
    assert len(res.retrieved_chunks) >= 1
