from __future__ import annotations

import argparse
import sys
from rag_router.core.types import Chunk
from rag_router.engine import AdaptiveRAGRouter


def main():
    parser = argparse.ArgumentParser(
        prog="rag-router",
        description="CLI for Agentic RAG Router: Adaptive & Speculative RAG Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Route command
    route_p = subparsers.add_parser("route", help="Analyze and route a query through the adaptive engine")
    route_p.add_argument("query", type=str, help="User query string")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run routing and speculative latency benchmarks")

    args = parser.parse_args()
    router = AdaptiveRAGRouter()

    # Index sample corpus
    router.index([
        Chunk(content="PyTorch 2.5 introduces native support for Float8 execution and TorchDynamo compile speedups."),
        Chunk(content="JAX uses pure functional transformations and XLA JIT compilation for maximum throughput."),
        Chunk(content="Error code ERR_503_SERVICE_UNAVAILABLE indicates downstream microservice outage."),
    ])

    if args.command == "route":
        res = router.route_and_execute(args.query)
        print("\n" + "=" * 60)
        print("Agentic RAG Router - Decision & Execution Report")
        print("=" * 60)
        print(f"Query:                 {res.query}")
        print(f"Detected Intent:       {res.decision.intent.value.upper()}")
        print(f"Recommended Strategy:  {res.decision.recommended_strategy.value}")
        print(f"Confidence:            {res.decision.confidence * 100:.1f}%")
        print(f"Reasoning:             {res.decision.reasoning}")
        if res.decision.sub_queries:
            print(f"Decomposed Sub-Queries: {', '.join(res.decision.sub_queries)}")
        print(f"\nRetrieved Chunks:      {len(res.retrieved_chunks)}")
        print(f"Grounding Score:       {res.grounding.grounding_score * 100:.1f}% ({res.grounding.hallucination_risk} Risk)")
        print(f"Total Latency:         {res.total_latency_ms} ms")
        print(f"\nAnswer:\n{res.answer}\n")

    elif args.command == "benchmark":
        import time
        print("Running 500 routing decisions benchmark...")
        queries = [
            "Hello there",
            "What is status for ERR_503_SERVICE_UNAVAILABLE?",
            "Compare PyTorch and JAX performance",
            "Explain TorchDynamo compile speedups",
        ]
        t0 = time.perf_counter()
        for i in range(500):
            q = queries[i % len(queries)]
            router.route_and_execute(q)
        total_time = time.perf_counter() - t0
        qps = 500 / total_time
        print(f"✓ Completed 500 adaptive routing decisions in {total_time:.3f}s ({qps:.1f} queries/sec).")


if __name__ == "__main__":
    main()
