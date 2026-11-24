from __future__ import annotations

import re
from typing import List


class SubQueryDecomposer:
    """
    Decomposes complex comparative or multi-attribute queries into targeted,
    independent sub-queries for parallel retrieval.
    """

    COMPARISON_REGEX = re.compile(
        r"(?:compare|contrast|difference\s+between|vs\.?)\s+([a-zA-Z0-9_\s]+?)\s+(?:and|versus|with)\s+([a-zA-Z0-9_\s]+)",
        re.IGNORECASE,
    )

    CONJUNCTION_REGEX = re.compile(
        r"(what\s+is\s+[a-zA-Z0-9_\s]+?)\s+and\s+(how\s+does\s+[a-zA-Z0-9_\s]+)",
        re.IGNORECASE,
    )

    def decompose(self, query: str, max_sub_queries: int = 3) -> List[str]:
        cleaned = query.strip()
        sub_queries: List[str] = []

        # 1. Comparative queries (e.g., "Compare X and Y")
        comp_match = self.COMPARISON_REGEX.search(cleaned)
        if comp_match:
            item_a, item_b = comp_match.groups()
            sub_queries.append(f"Features and specifications of {item_a.strip()}")
            sub_queries.append(f"Features and specifications of {item_b.strip()}")
            sub_queries.append(f"Direct comparison between {item_a.strip()} and {item_b.strip()}")
            return sub_queries[:max_sub_queries]

        # 2. Conjunction queries (e.g., "What is X and how does Y affect it?")
        conj_match = self.CONJUNCTION_REGEX.search(cleaned)
        if conj_match:
            part_a, part_b = conj_match.groups()
            sub_queries.append(part_a.strip())
            sub_queries.append(part_b.strip())
            return sub_queries[:max_sub_queries]

        # Fallback: single query
        return [cleaned]
