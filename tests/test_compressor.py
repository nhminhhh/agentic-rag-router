import pytest
from rag_router.compression.context_compressor import ContextCompressor
from rag_router.core.types import Chunk


def test_compressor_pruning():
    compressor = ContextCompressor(min_sentence_score=0.1)

    query = "Float8 compilation performance"
    chunks = [
        Chunk(
            id="c1",
            content="Welcome to our technical newsletter. Today is a great day. "
            "PyTorch 2.5 introduces native Float8 compilation performance enhancements. "
            "Remember to subscribe to our RSS feed.",
        ),
        Chunk(
            id="c2",
            content="Unrelated paragraph about baking cookies and sourdough bread. "
            "Another sentence discussing chocolate recipes.",
        ),
    ]

    compressed = compressor.compress_chunks(query, chunks, max_chars=300)

    assert "Float8 compilation performance enhancements" in compressed.compressed_text
    assert "baking cookies" not in compressed.compressed_text
    assert compressed.compression_ratio > 0.3
    assert compressed.retained_sentences_count >= 1


def test_compressor_empty_input():
    compressor = ContextCompressor()
    compressed = compressor.compress_chunks("hello", [])

    assert compressed.compressed_text == ""
    assert compressed.original_chars == 0
    assert compressed.compression_ratio == 0.0


def test_compressor_strict_budget():
    compressor = ContextCompressor(min_sentence_score=0.0)
    query = "architecture"
    chunks = [
        Chunk(id="c1", content="Architecture A is scalable. Architecture B is distributed. Architecture C is fast."),
    ]

    compressed = compressor.compress_chunks(query, chunks, max_chars=50)
    assert len(compressed.compressed_text) <= 50
