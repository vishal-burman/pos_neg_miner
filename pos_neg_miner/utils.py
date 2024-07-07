from typing import Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray


def get_normalized_embeddings(embeds: NDArray) -> NDArray:
    return embeds / np.linalg.norm(embeds)


def sanitate_text(text: str) -> str:
    return text.strip()


def validate_queries_and_candidates(queries: List[str], candidates: List[str]) -> Tuple[List[str], List[str]]:
    queries = list(map(sanitate_text, set(queries)))
    candidates = list(map(sanitate_text, set(candidates)))
    if not queries or not candidates:
        raise ValueError(f"Candidates or Queries cannot be zero!")

    return queries, candidates


def validate_queries_and_candidates_embeddings(queries_embeds: NDArray, candidates_embeds: NDArray) -> None:
    assert queries_embeds.ndim == 2, f"Embeddings should be 2-dimensional"
    assert candidates_embeds.ndim == 2, f"Embeddings should be 2-dimensional"
    assert (
        queries_embeds.shape[1] == candidates_embeds.shape[1]
    ), f"Queries and Candidate embeddings do not match: {queries_embeds.shape[1]}!={candidates_embeds.shape[1]}"
