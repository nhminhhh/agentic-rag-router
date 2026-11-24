from rag_router.classifier.intent_router import QueryIntentClassifier
from rag_router.core.types import QueryIntent, RetrievalStrategy


def test_intent_classification():
    router = QueryIntentClassifier()

    # Chitchat
    d1 = router.classify("Hello, how are you today?")
    assert d1.intent == QueryIntent.NO_RETRIEVAL
    assert d1.recommended_strategy == RetrievalStrategy.NONE

    # Exact code identifier
    d2 = router.classify("What is the status for `ERR_CONNECTION_REFUSED_503`?")
    assert d2.intent == QueryIntent.EXACT_LOOKUP
    assert d2.recommended_strategy == RetrievalStrategy.SPARSE_BM25

    # Multi-hop comparison
    d3 = router.classify("Compare the difference between PyTorch and JAX for training VLMs")
    assert d3.intent == QueryIntent.MULTI_HOP
    assert d3.recommended_strategy == RetrievalStrategy.MULTI_HOP_DECOMPOSED

    # General semantic query
    d4 = router.classify("Explain how FlashAttention reduces memory footprint")
    assert d4.intent == QueryIntent.HYBRID
    assert d4.recommended_strategy == RetrievalStrategy.HYBRID_RRF
