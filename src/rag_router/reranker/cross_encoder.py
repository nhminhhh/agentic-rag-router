from __future__ import annotations

import math
import re
from typing import List, Optional, Set, Tuple
from rag_router.core.types import Chunk


class CrossEncoderReranker:
    """
    Second-stage semantic cross-encoder reranker for retrieved candidate chunks.
    Evaluates joint (query, document) interaction with positional proximity,
    subword n-gram alignment, and term coverage.
    """

    STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "what", "how",
        "which", "who", "when", "where", "why", "does", "do", "did",
    }

    def __init__(
        self,
        coverage_weight: float = 0.45,
        proximity_weight: float = 0.35,
        subword_weight: float = 0.20,
    ):
        self.coverage_weight = coverage_weight
        self.proximity_weight = proximity_weight
        self.subword_weight = subword_weight

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text)]

    def _get_ngrams(self, word: str, n: int = 3) -> Set[str]:
        word = f"^{word}$"
        if len(word) < n:
            return {word}
        return {word[i : i + n] for i in range(len(word) - n + 1)}

    def _compute_proximity_bonus(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Computes positional proximity bonus: query tokens occurring near
        each other within doc receive higher weight.
        """
        if len(query_tokens) < 2 or not doc_tokens:
            return 0.0

        positions: List[Tuple[str, int]] = []
        q_set = set(query_tokens)
        for idx, token in enumerate(doc_tokens):
            if token in q_set:
                positions.append((token, idx))

        if len(positions) < 2:
            return 0.0

        # Measure min span covering multiple distinct query tokens
        found_tokens = {t for t, _ in positions}
        if len(found_tokens) < 2:
            return 0.2

        min_span = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, min(len(positions), i + 8)):
                if positions[i][0] != positions[j][0]:
                    span = abs(positions[j][1] - positions[i][1])
                    if span < min_span:
                        min_span = span

        if min_span == float("inf"):
            return 0.0

        # Exponential decay on token distance span
        return math.exp(-0.15 * max(0, min_span - 1))

    def score_pair(self, query: str, content: str) -> float:
        """Computes joint relevance score between query and chunk content."""
        q_tokens = [t for t in self._tokenize(query) if t not in self.STOPWORDS]
        doc_tokens = self._tokenize(content)

        if not q_tokens or not doc_tokens:
            return 0.0

        doc_set = set(doc_tokens)

        # 1. Term Coverage Score
        matched = sum(1 for t in q_tokens if t in doc_set)
        coverage_score = matched / len(q_tokens)

        # 2. Positional Proximity Bonus
        proximity_score = self._compute_proximity_bonus(q_tokens, doc_tokens)

        # 3. Subword N-Gram Overlap (robust to inflections and typos)
        q_ngrams: Set[str] = set()
        for t in q_tokens:
            q_ngrams.update(self._get_ngrams(t, n=3))

        doc_ngrams: Set[str] = set()
        for t in doc_tokens:
            doc_ngrams.update(self._get_ngrams(t, n=3))

        subword_score = 0.0
        if q_ngrams and doc_ngrams:
            intersection = len(q_ngrams.intersection(doc_ngrams))
            union = len(q_ngrams.union(doc_ngrams))
            subword_score = intersection / union if union > 0 else 0.0

        # Weighted aggregate score
        raw_score = (
            self.coverage_weight * coverage_score
            + self.proximity_weight * proximity_score
            + self.subword_weight * subword_score
        )

        return round(min(1.0, max(0.0, raw_score)), 4)

    def rerank(self, query: str, chunks: List[Chunk], top_k: Optional[int] = None) -> List[Chunk]:
        """Reranks candidate chunks by cross-encoder relevance score."""
        if not chunks:
            return []

        scored_chunks: List[Tuple[float, Chunk]] = []
        for chunk in chunks:
            s = self.score_pair(query, chunk.content)
            # Store updated rerank score into chunk metadata
            chunk.metadata["rerank_score"] = s
            scored_chunks.append((s, chunk))

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        ranked = [c for _, c in scored_chunks]
        if top_k is not None:
            return ranked[:top_k]
        return ranked
