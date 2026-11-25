import pytest
from rag_router.core.types import Chunk
from rag_router.reranker.cross_encoder import CrossEncoderReranker


def test_cross_encoder_scoring_and_ranking():
    reranker = CrossEncoderReranker()

    query = "PyTorch compile float8 execution speedup"
    chunks = [
        Chunk(id="c1", content="General machine learning models require careful hyperparameter tuning."),
        Chunk(id="c2", content="PyTorch 2.5 introduces native Float8 compile execution for 2x speedup."),
        Chunk(id="c3", content="Execution speedup in unrelated C++ graphics pipeline."),
    ]

    reranked = reranker.rerank(query, chunks, top_k=2)

    assert len(reranked) == 2
    # c2 must rank highest because it covers all key terms with high proximity
    assert reranked[0].id == "c2"
    assert reranked[0].metadata["rerank_score"] > reranked[1].metadata["rerank_score"]


def test_cross_encoder_proximity_bonus():
    reranker = CrossEncoderReranker()
    query = "database latency"

    # Close proximity
    doc_near = "High database latency causes HTTP 504 errors in production."
    # Dispersed tokens
    doc_far = "Our database manages user profiles and later in the architecture we measure network latency."

    score_near = reranker.score_pair(query, doc_near)
    score_far = reranker.score_pair(query, doc_far)

    assert score_near > score_far


def test_cross_encoder_empty_inputs():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("any query", []) == []
    assert reranker.score_pair("", "some content") == 0.0
