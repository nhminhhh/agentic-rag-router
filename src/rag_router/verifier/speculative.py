from __future__ import annotations

import re
from typing import List, Optional, Tuple
from rag_router.core.types import Chunk, GroundingResult
from .grounding import GroundingVerifier


class SpeculativeDraftEngine:
    """
    Constructs speculative extractive drafts directly from top-ranked chunks
    and validates grounding, providing sub-millisecond responses when confidence is high.
    """

    def __init__(self, verifier: Optional[GroundingVerifier] = None):
        self.verifier = verifier or GroundingVerifier()

    def generate_and_verify(
        self,
        query: str,
        chunks: List[Chunk],
    ) -> Tuple[Optional[str], GroundingResult]:
        if not chunks:
            return None, GroundingResult(is_grounded=False, grounding_score=0.0, hallucination_risk="HIGH")

        # Extract top sentence that most overlaps query
        q_words = set(re.findall(r"\w+", query.lower()))
        best_sentence = ""
        best_overlap = 0

        for chunk in chunks[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
            for s in sentences:
                s_words = set(re.findall(r"\w+", s.lower()))
                overlap = len(q_words.intersection(s_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = s.strip()

        if not best_sentence:
            best_sentence = chunks[0].content.strip()

        # Validate speculative draft against full retrieved context
        grounding = self.verifier.verify(best_sentence, chunks)
        if grounding.is_grounded and grounding.grounding_score >= 0.70:
            return best_sentence, grounding

        return None, grounding
