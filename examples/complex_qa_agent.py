from __future__ import annotations

import time
from typing import List
from rag_router.core.config import RouterConfig
from rag_router.core.types import Chunk
from rag_router.engine import AdaptiveRAGRouter


class AutonomousTechQAAgent:
    def __init__(self):
        config = RouterConfig(
            min_grounding_threshold=0.6,
            enable_speculative_draft=True,
            max_sub_queries=3,
            top_k=3,
        )
        self.router = AdaptiveRAGRouter(config=config)

    def ingest_knowledge_base(self, docs: List[dict]):
        chunks = [
            Chunk(
                id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            for doc in docs
        ]
        self.router.index(chunks)

    def answer_question(self, user_question: str) -> None:
        print("=" * 60)
        print(f"Question: {user_question}")
        response = self.router.route_and_execute(user_question)
        print(f"Intent:   {response.decision.intent.value}")
        print(f"Strategy: {response.decision.recommended_strategy.value}")
        print(f"Answer:   {response.answer}")
        print(f"Latency:  {response.total_latency_ms:.2f} ms")


def main():
    agent = AutonomousTechQAAgent()
    agent.ingest_knowledge_base([
        {
            "id": "arch-01",
            "content": "Engine-A optimizes memory allocation with partitioned page blocks.",
            "metadata": {"category": "architecture"},
        },
        {
            "id": "arch-02",
            "content": "Engine-B employs prefix index caching for high concurrency streams.",
            "metadata": {"category": "architecture"},
        },
    ])

    agent.answer_question("Compare Engine-A memory allocation versus Engine-B prefix index caching")
    agent.answer_question("Hello assistant")


if __name__ == "__main__":
    main()
