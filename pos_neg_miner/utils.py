from typing import List, Tuple


def sanitate_text(text: str) -> str:
    return text.strip()


def validate_queries_and_candidates(
    queries: List[str], candidates: List[str]
) -> Tuple[List[str], List[str]]:
    queries = list(map(sanitate_text, set(queries)))
    candidates = list(map(sanitate_text, set(candidates)))
    if not queries or not candidates:
        raise ValueError(f"Candidates or Queries cannot be zero!")

    return queries, candidates
