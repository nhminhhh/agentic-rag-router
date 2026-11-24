from __future__ import annotations

from typing import Dict, List
from rag_router.core.types import Chunk
from .base import BaseRetrievalStrategy
from .dense import DenseVectorStrategy
from .sparse import SparseBM25Strategy


class ReciprocalRankFusionStrategy(BaseRetrievalStrategy):
    """
    Combines sparse lexical BM25 and dense semantic retrieval using Reciprocal Rank Fusion:
        RRF_Score(d) = 1 / (k + rank_sparse(d)) + 1 / (k + rank_dense(d))
    """

    def __init__(self, rrf_k: int = 60, dense_dimension: int = 128):
        self.rrf_k = rrf_k
        self.sparse = SparseBM25Strategy()
        self.dense = DenseVectorStrategy(dimension=dense_dimension)
        self.all_chunks: Dict[str, Chunk] = {}

    def index(self, chunks: List[Chunk]) -> None:
        self.all_chunks = {c.id: c for c in chunks}
        self.sparse.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        # Retrieve candidate pools from both branches
        sparse_hits = self.sparse.retrieve(query, top_k=top_k * 2)
        dense_hits = self.dense.retrieve(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}

        # Accumulate sparse ranks
        for rank, chunk in enumerate(sparse_hits, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Accumulate dense ranks
        for rank, chunk in enumerate(dense_hits, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (self.rrf_k + rank))

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[Chunk] = []

        for chk_id, score in ranked:
            c = self.all_chunks[chk_id].model_copy()
            c.score = round(score, 6)
            results.append(c)

        return results
