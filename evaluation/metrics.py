"""Ranking metrics for the matching evaluation.

`ranked` is the list of job ids in the order the system returned them,
`relevant` is the set of job ids that count as relevant for that resume.
"""


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    raise NotImplementedError


def mrr(ranked: list[str], relevant: set[str]) -> float:
    raise NotImplementedError


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    raise NotImplementedError
