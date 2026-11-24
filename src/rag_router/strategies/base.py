from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
from rag_router.core.types import Chunk


class BaseRetrievalStrategy(ABC):
    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        pass
