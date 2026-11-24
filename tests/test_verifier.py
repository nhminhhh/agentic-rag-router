from rag_router.core.types import Chunk
from rag_router.verifier.grounding import GroundingVerifier
from rag_router.verifier.speculative import SpeculativeDraftEngine


def test_grounding_verifier():
    verifier = GroundingVerifier()
    chunks = [
        Chunk(content="PyTorch 2.5 introduces native support for Float8 execution and TorchDynamo compile speedups."),
    ]

    # Grounded answer
    grounded_ans = "PyTorch 2.5 supports native Float8 execution."
    res1 = verifier.verify(grounded_ans, chunks)
    assert res1.is_grounded is True
    assert res1.hallucination_risk == "LOW"
    assert len(res1.unsupported_claims) == 0

    # Hallucinated answer
    hallucinated_ans = "PyTorch was created by NASA to fly Apollo spacecraft to Mars in 1969."
    res2 = verifier.verify(hallucinated_ans, chunks)
    assert res2.is_grounded is False
    assert res2.hallucination_risk in ("MEDIUM", "HIGH")
    assert len(res2.unsupported_claims) >= 1


def test_speculative_draft_engine():
    engine = SpeculativeDraftEngine()
    chunks = [
        Chunk(content="FlashAttention reduces attention memory complexity from quadratic O(N^2) to linear O(N)."),
    ]

    draft, grounding = engine.generate_and_verify("What is the memory complexity of FlashAttention?", chunks)
    assert draft is not None
    assert "linear O(N)" in draft
    assert grounding.is_grounded is True
