from typing import Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray


def get_normalized_embeddings(embeds: NDArray) -> NDArray:
    """Normalize embeddings to unit scale so that dot-product range lies in [-1, 1]
    Args:
        embeds: Numpy embeddings to be normalized.

    Returns:
        NDArray: Normalized numpy embeddings
    """
    return embeds / np.linalg.norm(embeds)


def sanitate_text(text: str) -> str:
    """Sanitized a piece of text for further validation process.

    Args:
        text: The text string to be sanitized.

    Returns:
        str: The sanitized string.
    """
    return text.strip()


def validate_queries_and_candidates(queries: List[str], candidates: List[str]) -> Tuple[List[str], List[str]]:
    """Sanitizes and validates lists of queries and candidates.

    This function takes in two lists of strings, sanitizes each element by removing duplicates and
    applying a sanitation function, and then checks that neither list is empty. If either list is
    empty after sanitization, it raises a ValueError.

    Args:
        queries (List[str]): A list of query strings to be sanitized and validated.
        candidates (List[str]): A list of candidate strings to be sanitized and validated.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the sanitized lists of queries and candidates.

    Raises:
        ValueError: If either the queries or candidates list is empty after sanitization.
    """
    queries = list(map(sanitate_text, set(queries)))
    candidates = list(map(sanitate_text, set(candidates)))
    if not queries or not candidates:
        raise ValueError(f"Candidates or Queries cannot be zero!")

    return queries, candidates


def validate_queries_and_candidates_embeddings(queries_embeds: NDArray, candidates_embeds: NDArray) -> None:
    """Validate shape and dimensions of query and candidates embeddings.

    Args:
        queries_embeds: Numpy array of queries embeddings.
        candidates_embeds: Numpy array of candidates embeddings.
    """
    assert queries_embeds.ndim == 2, f"Embeddings should be 2-dimensional"
    assert candidates_embeds.ndim == 2, f"Embeddings should be 2-dimensional"
    assert (
        queries_embeds.shape[1] == candidates_embeds.shape[1]
    ), f"Queries and Candidate embeddings do not match: {queries_embeds.shape[1]}!={candidates_embeds.shape[1]}"
