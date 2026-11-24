from rag_router.core.types import Chunk, QueryIntent, RetrievalStrategy
from rag_router.engine import AdaptiveRAGRouter


def test_adaptive_rag_router_lifecycle():
    router = AdaptiveRAGRouter()
    
    chunks = [
        Chunk(content="PyTorch uses dynamic computational graphs (eager mode) for flexible debugging."),
        Chunk(content="JAX uses pure functional transformations and XLA JIT compilation for maximum throughput."),
        Chunk(content="Error code 0x80070005 indicates access denied permissions on Windows subsystem."),
    ]
    router.index(chunks)

    # 1. Chitchat -> NO_RETRIEVAL
    res1 = router.route_and_execute("Hello, what's up?")
    assert res1.decision.intent == QueryIntent.NO_RETRIEVAL
    assert len(res1.retrieved_chunks) == 0
    assert res1.grounding.is_grounded is True

    # 2. Exact code lookup -> SPARSE_BM25
    res2 = router.route_and_execute("What does error 0x80070005 mean?")
    assert res2.decision.intent == QueryIntent.EXACT_LOOKUP
    assert len(res2.retrieved_chunks) >= 1
    assert any("0x80070005" in c.content for c in res2.retrieved_chunks)

    # 3. Multi-hop comparison -> MULTI_HOP_DECOMPOSED
    res3 = router.route_and_execute("Compare PyTorch and JAX computational compilation models")
    assert res3.decision.intent == QueryIntent.MULTI_HOP
    assert len(res3.decision.sub_queries) >= 2
    assert len(res3.retrieved_chunks) >= 2
    assert res3.grounding.is_grounded is True
