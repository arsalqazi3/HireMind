"""Ranking metrics for the matching evaluation.

`ranked` is the list of job ids in the order the system returned them,
`relevant` is the set of job ids that count as relevant for that resume.
A job id repeated in `ranked` only counts the first time, so a system cannot
raise its score by returning the same job twice. All metrics are between 0 and 1.
"""
import math


def _unique(ranked: list[str]) -> list[str]:
    return list(dict.fromkeys(ranked))


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Share of the k slots that hold a relevant job. Empty slots count as misses."""
    return sum(job in relevant for job in _unique(ranked)[:k]) / k


def mrr(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant job, or 0 if there is none.

    Averaged over resumes this is the mean reciprocal rank.
    """
    for rank, job in enumerate(_unique(ranked), start=1):
        if job in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Normalized discounted cumulative gain with yes or no relevance."""
    dcg = sum(1 / math.log2(rank + 1) for rank, job in enumerate(_unique(ranked)[:k], start=1) if job in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0
