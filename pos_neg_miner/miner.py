from typing import List
from utils import validate_queries_and_candidates

class Miner:
    def __init__(self,):
        pass
    
    def __call__(self, queries: List[str], candidates: List[str]):
        queries, candidates = validate_queries_and_candidates(queries=queries, candidates=candidates)
