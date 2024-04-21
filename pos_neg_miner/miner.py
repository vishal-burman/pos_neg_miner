from typing import Callable, List

from .utils import (
    validate_queries_and_candidates,
    validate_queries_and_candidates_embeddings,
    get_normalized_embeddings
)


class Miner:
    def __init__(
        self,
    ):
        pass

    def __call__(self,
                 queries: List[str],
                 candidates: List[str],
                 embedder: Callable,
                 normalize_embeddings: bool):
        queries, candidates = validate_queries_and_candidates(
            queries=queries, candidates=candidates
        )
        queries_embeds = embedder(queries)
        candidates_embeds = embedder(candidates)
        validate_queries_and_candidates_embeddings(
            queries_embeds=queries_embeds, candidates_embeds=candidates_embeds
        )
        if normalize_embeddings:
            queries_embeds, candidates_embeds = get_normalized_embeddings(queries_embeds), get_normalized_embeddings(candidates_embeds)
        
        return queries_embeds, candidates_embeds