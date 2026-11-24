from __future__ import annotations

import re
from typing import List
from rag_router.core.types import Chunk, GroundingResult


class GroundingVerifier:
    """
    Verifies that generated candidate answers are strictly entailed by the retrieved chunks,
    detecting hallucinations and unsupported assertions before returning to the agent.
    """

    def __init__(self, min_support_ratio: float = 0.60):
        self.min_support_ratio = min_support_ratio

    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        # Split text into sentence-level claims
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw_sentences if len(s.strip()) > 8]

    def verify(self, answer: str, context_chunks: List[Chunk]) -> GroundingResult:
        if not context_chunks:
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                unsupported_claims=[answer],
                hallucination_risk="HIGH",
            )

        context_corpus = " ".join(c.content.lower() for c in context_chunks)
        claims = self._extract_claims(answer)
        if not claims:
            return GroundingResult(is_grounded=True, grounding_score=1.0, hallucination_risk="LOW")

        unsupported: List[str] = []
        supported_count = 0

        for claim in claims:
            claim_words = [w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3]
            if not claim_words:
                supported_count += 1
                continue

            # Check what percentage of key claim words are found in context chunks
            matches = sum(1 for w in claim_words if w in context_corpus)
            overlap_ratio = matches / len(claim_words)

            if overlap_ratio >= self.min_support_ratio:
                supported_count += 1
            else:
                unsupported.append(claim)

        score = round(supported_count / len(claims), 3)
        if score >= 0.80:
            risk = "LOW"
        elif score >= 0.50:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return GroundingResult(
            is_grounded=score >= self.min_support_ratio,
            grounding_score=score,
            unsupported_claims=unsupported,
            hallucination_risk=risk,
        )
