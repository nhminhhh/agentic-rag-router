from __future__ import annotations

import re
from typing import List, NamedTuple, Set
from rag_router.core.types import Chunk


class CompressedContext(NamedTuple):
    compressed_text: str
    original_chars: int
    compressed_chars: int
    compression_ratio: float
    retained_sentences_count: int


class ContextCompressor:
    """
    Intelligent context compressor for Agentic RAG.
    Extracts high-salience sentences matching query intents and eliminates
    redundant context to minimize LLM prompt token consumption.
    """

    STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "what", "how",
        "which", "who", "when", "where", "why", "does", "do", "did",
    }

    def __init__(self, min_sentence_score: float = 0.15):
        self.min_sentence_score = min_sentence_score

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence terminals while keeping punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> Set[str]:
        return {
            t.lower()
            for t in re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text)
            if t.lower() not in self.STOPWORDS
        }

    def _score_sentence(self, sentence: str, query_tokens: Set[str]) -> float:
        s_tokens = self._tokenize(sentence)
        if not s_tokens or not query_tokens:
            return 0.0

        overlap = len(s_tokens.intersection(query_tokens))
        term_ratio = overlap / len(query_tokens)

        # Bonus for presence of technical identifiers or numbers
        has_numbers = bool(re.search(r"\d+", sentence))
        num_bonus = 0.15 if has_numbers else 0.0

        # Length penalty for tiny fragments (< 15 chars)
        if len(sentence) < 15:
            return 0.05

        return min(1.0, term_ratio * 0.85 + num_bonus)

    def compress_chunks(
        self,
        query: str,
        chunks: List[Chunk],
        max_chars: int = 1200,
    ) -> CompressedContext:
        """
        Compresses retrieved chunks against query terms within max_chars budget.
        """
        if not chunks:
            return CompressedContext(
                compressed_text="",
                original_chars=0,
                compressed_chars=0,
                compression_ratio=0.0,
                retained_sentences_count=0,
            )

        q_tokens = self._tokenize(query)
        total_original_chars = sum(len(c.content) for c in chunks)

        scored_sentences = []
        for chunk in chunks:
            sentences = self._split_sentences(chunk.content)
            for s in sentences:
                score = self._score_sentence(s, q_tokens)
                if score >= self.min_sentence_score:
                    scored_sentences.append((score, s, chunk.id))

        # Sort by salience score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        selected: List[str] = []
        current_len = 0
        seen_texts: Set[str] = set()

        for score, s, chunk_id in scored_sentences:
            if s in seen_texts:
                continue
            if current_len + len(s) + 1 <= max_chars:
                selected.append(s)
                seen_texts.add(s)
                current_len += len(s) + 1
            else:
                break

        # If no sentences passed threshold, fallback to first chunk snippet
        if not selected and chunks:
            fallback = chunks[0].content[:max_chars]
            selected.append(fallback)
            current_len = len(fallback)

        compressed_text = " ".join(selected)
        compressed_len = len(compressed_text)
        ratio = (
            (total_original_chars - compressed_len) / total_original_chars
            if total_original_chars > 0
            else 0.0
        )

        return CompressedContext(
            compressed_text=compressed_text,
            original_chars=total_original_chars,
            compressed_chars=compressed_len,
            compression_ratio=round(max(0.0, ratio), 4),
            retained_sentences_count=len(selected),
        )
