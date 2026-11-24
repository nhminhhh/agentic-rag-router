from pydantic import BaseModel, Field


class RouterConfig(BaseModel):
    min_grounding_threshold: float = Field(default=0.65, description="Minimum grounding score before self-reflection trigger")
    enable_speculative_draft: bool = Field(default=True, description="Enable speculative low-latency drafting")
    dense_dimension: int = Field(default=128, description="Hashed dense embedding dimension")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing constant")
    max_sub_queries: int = Field(default=3, description="Maximum sub-queries for multi-hop decomposition")
    top_k: int = Field(default=4, description="Default number of chunks to retrieve")
