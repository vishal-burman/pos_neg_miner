from typing import Callable, List

from utils import (
    validate_queries_and_candidates,
    validate_queries_and_candidates_embeddings,
)


class Miner:
    def __init__(
        self,
    ):
        pass

    def __call__(self, queries: List[str], candidates: List[str], embedder: Callable):
        queries, candidates = validate_queries_and_candidates(
            queries=queries, candidates=candidates
        )
        queries_embeds = embedder(queries)
        candidates_embeds = embedder(candidates)
        validate_queries_and_candidates_embeddings(
            queries_embeds=queries_embeds, candidates_embeds=candidates_embeds
        )
