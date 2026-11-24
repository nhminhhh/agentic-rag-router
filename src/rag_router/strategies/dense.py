from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List
from rag_router.core.types import Chunk
from .base import BaseRetrievalStrategy


class DenseVectorStrategy(BaseRetrievalStrategy):
    """
    Hashed subword dense semantic representation for zero-dependency vector retrieval.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        clean = f"_{text.lower().strip()}_"
        words = re.findall(r"\w+", clean)

        for w in words:
            h = hashlib.md5(w.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dimension
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += 1.5 * sign

        for n in (3, 4, 5):
            for i in range(len(clean) - n + 1):
                gram = clean[i : i + n]
                h = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dimension
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                vec[idx] += 1.0 * sign

        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0.0 else vec

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.embeddings = {c.id: self._embed(c.content) for c in chunks}

    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        if not self.chunks:
            return []

        q_vec = self._embed(query)
        scores: Dict[str, float] = {}

        for chk_id, doc_vec in self.embeddings.items():
            dot = sum(a * b for a, b in zip(q_vec, doc_vec))
            scores[chk_id] = max(0.0, dot)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[Chunk] = []
        for chk_id, sim in ranked:
            c = self.chunks[chk_id].model_copy()
            c.score = round(sim, 4)
            results.append(c)

        return results
