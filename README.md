# Agentic RAG Router

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Pydantic: v2](https://img.shields.io/badge/Pydantic-v2-orange.svg)](https://docs.pydantic.dev/)

**Agentic RAG Router** is an adaptive, speculative retrieval-augmented generation framework designed for autonomous LLM agents.

---

## 🚀 Key Pillars

1. **Query Intent Routing**: Heuristic zero-latency classification (No Retrieval, Sparse BM25, Dense Semantic, Multi-Hop Decomposed, Hybrid RRF).
2. **Multi-Strategy Retrieval**: Inverted index BM25, dense subword embeddings, and Reciprocal Rank Fusion.
3. **Speculative Grounding & Verification**: Validates claims before final answer generation to eliminate hallucinations.
4. **Multi-Hop Query Decomposition**: Splits complex comparative inquiries into targeted parallel sub-queries.
