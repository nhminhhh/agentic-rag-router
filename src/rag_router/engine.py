from __future__ import annotations

import time
from typing import List, Optional
from rag_router.classifier.intent_router import QueryIntentClassifier
from rag_router.classifier.query_decomposer import SubQueryDecomposer
from rag_router.core.config import RouterConfig
from rag_router.core.types import (
    Chunk,
    GroundingResult,
    QueryIntent,
    RAGResponse,
    RetrievalStrategy,
    RouterDecision,
)
from rag_router.strategies.dense import DenseVectorStrategy
from rag_router.strategies.hybrid import ReciprocalRankFusionStrategy
from rag_router.strategies.sparse import SparseBM25Strategy
from rag_router.verifier.grounding import GroundingVerifier
from rag_router.verifier.speculative import SpeculativeDraftEngine


class AdaptiveRAGRouter:
    """
    Production-grade Adaptive & Speculative RAG engine.
    Orchestrates intent classification, strategy routing, sub-query decomposition,
    speculative draft verification, and hallucination guardrails.
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self.classifier = QueryIntentClassifier()
        self.decomposer = SubQueryDecomposer()

        # Strategies
        self.sparse_strategy = SparseBM25Strategy()
        self.dense_strategy = DenseVectorStrategy(dimension=self.config.dense_dimension)
        self.hybrid_strategy = ReciprocalRankFusionStrategy(
            rrf_k=self.config.rrf_k,
            dense_dimension=self.config.dense_dimension,
        )

        # Verifiers
        self.grounding_verifier = GroundingVerifier(min_support_ratio=self.config.min_grounding_threshold)
        self.speculative_engine = SpeculativeDraftEngine(verifier=self.grounding_verifier)

        self._indexed_chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        """Indexes corpus across all strategy backends."""
        self._indexed_chunks = list(chunks)
        self.sparse_strategy.index(chunks)
        self.dense_strategy.index(chunks)
        self.hybrid_strategy.index(chunks)

    def route_and_execute(self, query: str) -> RAGResponse:
        t_start = time.perf_counter()

        # Step 1: Intent Classification
        decision = self.classifier.classify(query)

        # Step 2: Handle No Retrieval
        if decision.intent == QueryIntent.NO_RETRIEVAL:
            latency = (time.perf_counter() - t_start) * 1000.0
            return RAGResponse(
                query=query,
                answer="Direct response: This query does not require external retrieval.",
                retrieved_chunks=[],
                decision=decision,
                grounding=GroundingResult(is_grounded=True, grounding_score=1.0, hallucination_risk="LOW"),
                total_latency_ms=round(latency, 2),
            )

        # Step 3: Multi-Hop Decomposition or Direct Routing
        retrieved_chunks: List[Chunk] = []

        if decision.intent == QueryIntent.MULTI_HOP:
            sub_queries = self.decomposer.decompose(query, max_sub_queries=self.config.max_sub_queries)
            decision.sub_queries = sub_queries

            seen_ids = set()
            for sub_q in sub_queries:
                sub_hits = self.hybrid_strategy.retrieve(sub_q, top_k=2)
                for h in sub_hits:
                    if h.id not in seen_ids:
                        retrieved_chunks.append(h)
                        seen_ids.add(h.id)
        elif decision.intent == QueryIntent.EXACT_LOOKUP:
            retrieved_chunks = self.sparse_strategy.retrieve(query, top_k=self.config.top_k)
        else:
            # Hybrid RRF default
            retrieved_chunks = self.hybrid_strategy.retrieve(query, top_k=self.config.top_k)

        # Step 4: Speculative Draft Generation & Grounding Check
        answer = ""
        grounding = GroundingResult(is_grounded=True, grounding_score=1.0, hallucination_risk="LOW")

        if self.config.enable_speculative_draft and retrieved_chunks:
            draft, g_res = self.speculative_engine.generate_and_verify(query, retrieved_chunks)
            if draft:
                answer = draft
                grounding = g_res
            else:
                answer = f"Synthesized answer based on {len(retrieved_chunks)} retrieved contexts."
                grounding = g_res
        else:
            answer = f"Retrieved {len(retrieved_chunks)} relevant reference chunks."

        latency = (time.perf_counter() - t_start) * 1000.0
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            decision=decision,
            grounding=grounding,
            total_latency_ms=round(latency, 2),
        )
