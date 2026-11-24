from rag_router.core.types import Chunk
from rag_router.strategies.dense import DenseVectorStrategy
from rag_router.strategies.hybrid import ReciprocalRankFusionStrategy
from rag_router.strategies.sparse import SparseBM25Strategy


def test_sparse_bm25_exact_match():
    bm25 = SparseBM25Strategy()
    c1 = Chunk(content="Error code ERR_503_SERVICE_UNAVAILABLE occurred in checkout service")
    c2 = Chunk(content="Payment failed due to timeout in billing service")
    bm25.index([c1, c2])

    hits = bm25.retrieve("ERR_503_SERVICE_UNAVAILABLE", top_k=1)
    assert len(hits) == 1
    assert hits[0].id == c1.id
    assert hits[0].score > 0.0


def test_dense_vector_semantic_search():
    dense = DenseVectorStrategy()
    c1 = Chunk(content="Optimizing deep learning neural network architectures using PyTorch")
    c2 = Chunk(content="Baking sourdough bread with organic wheat flour")
    dense.index([c1, c2])

    hits = dense.retrieve("machine learning models PyTorch", top_k=1)
    assert len(hits) == 1
    assert hits[0].id == c1.id


def test_reciprocal_rank_fusion():
    rrf = ReciprocalRankFusionStrategy()
    c1 = Chunk(content="FlashAttention V3 kernel optimization in PyTorch")
    c2 = Chunk(content="General hardware overview for cloud instances")
    rrf.index([c1, c2])

    hits = rrf.retrieve("FlashAttention PyTorch kernel", top_k=1)
    assert len(hits) == 1
    assert hits[0].id == c1.id
