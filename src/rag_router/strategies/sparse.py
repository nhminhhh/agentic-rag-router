from __future__ import annotations

import math
import re
from typing import Dict, List, Set
from rag_router.core.types import Chunk
from .base import BaseRetrievalStrategy


class SparseBM25Strategy(BaseRetrievalStrategy):
    """
    In-memory Okapi BM25 implementation for lexical keyword search, exact phrases, and SKU lookups.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: Dict[str, Chunk] = {}
        self.doc_lens: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.inverted_index: Dict[str, Dict[str, int]] = {}  # term -> doc_id -> count

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks.clear()
        self.doc_lens.clear()
        self.doc_freqs.clear()
        self.inverted_index.clear()

        total_words = 0
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            tokens = self._tokenize(chunk.content)
            self.doc_lens[chunk.id] = len(tokens)
            total_words += len(tokens)

            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1

            for term, count in counts.items():
                self.inverted_index.setdefault(term, {})[chunk.id] = count
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = (total_words / len(chunks)) if chunks else 1.0

    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        query_terms = self._tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scores: Dict[str, float] = {chk_id: 0.0 for chk_id in self.chunks}
        n_docs = len(self.chunks)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            df = self.doc_freqs.get(term, 0)
            # Standard Robertson-Spärck Jones IDF
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in self.inverted_index[term].items():
                doc_len = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (tf * (self.k1 + 1.0)) / max(1e-5, denom)
                scores[doc_id] += term_score

        ranked_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        results: List[Chunk] = []
        for doc_id in ranked_ids[:top_k]:
            if scores[doc_id] > 0.0:
                c = self.chunks[doc_id].model_copy()
                c.score = round(scores[doc_id], 4)
                results.append(c)

        return results
