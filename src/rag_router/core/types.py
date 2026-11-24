from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    NO_RETRIEVAL = "no_retrieval"      # Chitchat, greetings, direct arithmetic, general knowledge
    EXACT_LOOKUP = "exact_lookup"      # Code identifiers, IDs, hashes, SKUs, exact phrases
    CONCEPTUAL = "conceptual"          # Semantic questions, summaries, high-level explanations
    MULTI_HOP = "multi_hop"            # Complex comparative questions needing sub-query decomposition
    HYBRID = "hybrid"                  # Mixed keyword + semantic


class RetrievalStrategy(str, Enum):
    NONE = "none"
    SPARSE_BM25 = "sparse_bm25"
    DENSE_VECTOR = "dense_vector"
    HYBRID_RRF = "hybrid_rrf"
    MULTI_HOP_DECOMPOSED = "multi_hop_decomposed"


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:8]}")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class RouterDecision(BaseModel):
    query: str
    intent: QueryIntent
    recommended_strategy: RetrievalStrategy
    confidence: float
    sub_queries: List[str] = Field(default_factory=list)
    reasoning: str


class GroundingResult(BaseModel):
    is_grounded: bool
    grounding_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    hallucination_risk: str = "LOW"  # LOW, MEDIUM, HIGH


class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    decision: RouterDecision
    grounding: GroundingResult
    total_latency_ms: float = 0.0
