from __future__ import annotations

import re
from typing import List, Tuple
from rag_router.core.types import QueryIntent, RetrievalStrategy, RouterDecision


class QueryIntentClassifier:
    """
    Lightweight heuristic classifier routing queries to the most cost- and latency-efficient
    retrieval strategy without requiring slow external LLM judge roundtrips.
    """

    CHITCHAT_PATTERNS = [
        r"^(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
        r"^(how\s+are\s+you|who\s+are\s+you|what\s+can\s+you\s+do)\b",
        r"^(thank\s+you|thanks|bye|goodbye)\b",
        r"^\d+\s*[\+\-\*\/]\s*\d+\s*=?$",  # Pure arithmetic e.g. "12 + 45"
    ]

    EXACT_LOOKUP_PATTERNS = [
        r"\b(sku|id|hash|sha256|uuid|err_code|status_code|0x[0-9a-fA-F]+)\b",
        r"\b(?=[a-zA-Z0-9_\-]*[0-9])(?=[a-zA-Z0-9_\-]*[a-zA-Z])[a-zA-Z0-9_\-]{10,}\b",  # Mixed alphanumeric IDs/hashes
        r"`[^`]+`",  # Code tokens enclosed in backticks
        r'\"[^\"]+\"',  # Exact quoted phrases
    ]

    MULTI_HOP_PATTERNS = [
        r"\b(compare|contrast|difference\s+between|versus|vs\.?)\b",
        r"\b(how\s+does\s+.+\s+relate\s+to\s+.+)\b",
        r"\b(which\s+one\s+is\s+better|pros\s+and\s+cons\s+of)\b",
        r"\b(both\s+.+\s+and\s+.+)\b",
    ]

    def classify(self, query: str) -> RouterDecision:
        cleaned = query.strip()

        # 1. Check No Retrieval (Chitchat / Direct Arithmetic)
        for pat in self.CHITCHAT_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return RouterDecision(
                    query=query,
                    intent=QueryIntent.NO_RETRIEVAL,
                    recommended_strategy=RetrievalStrategy.NONE,
                    confidence=0.95,
                    reasoning="Query is conversational chitchat or trivial math; retrieval not needed.",
                )

        # 2. Check Multi-Hop Complex Reasoning
        for pat in self.MULTI_HOP_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return RouterDecision(
                    query=query,
                    intent=QueryIntent.MULTI_HOP,
                    recommended_strategy=RetrievalStrategy.MULTI_HOP_DECOMPOSED,
                    confidence=0.88,
                    reasoning="Query requires multi-hop comparative synthesis across distinct topics.",
                )

        # 3. Check Exact / SKU / Code lookup
        for pat in self.EXACT_LOOKUP_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return RouterDecision(
                    query=query,
                    intent=QueryIntent.EXACT_LOOKUP,
                    recommended_strategy=RetrievalStrategy.SPARSE_BM25,
                    confidence=0.90,
                    reasoning="Query contains exact identifiers or quoted keywords suitable for inverted index sparse match.",
                )

        # 4. Default to Hybrid RRF for general conceptual inquiries
        return RouterDecision(
            query=query,
            intent=QueryIntent.HYBRID,
            recommended_strategy=RetrievalStrategy.HYBRID_RRF,
            confidence=0.85,
            reasoning="Query benefits from combined dense semantic search and sparse lexical ranking.",
        )
