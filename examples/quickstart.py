from __future__ import annotations

from rag_router.core.types import Chunk
from rag_router.engine import AdaptiveRAGRouter


def run_quickstart():
    router = AdaptiveRAGRouter()
    documents = [
        Chunk(id="c1", content="Module-A provides asynchronous tensor compilation and kernel fusion."),
        Chunk(id="c2", content="Module-B executes distributed communication across worker nodes."),
        Chunk(id="c3", content="Status code ERR_CONN_TIMEOUT occurs when downstream latency exceeds 5000ms."),
    ]
    router.index(documents)

    queries = [
        "Hello there",
        "Explain ERR_CONN_TIMEOUT issue",
        "Compare Module-A kernel fusion versus Module-B distributed communication",
    ]

    for q in queries:
        res = router.route_and_execute(q)
        print(f"Query: {q}")
        print(f"Intent: {res.decision.intent.value} | Strategy: {res.decision.recommended_strategy.value}")
        print(f"Answer: {res.answer}\n")


if __name__ == "__main__":
    run_quickstart()
