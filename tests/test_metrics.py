import math

import pytest

from evaluation.metrics import mrr, ndcg_at_k, precision_at_k


def test_precision_at_k():
    assert precision_at_k(["a", "x", "b", "y", "z"], {"a", "b"}) == pytest.approx(2 / 5)
    assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(2 / 5)  # empty slots are misses
    assert precision_at_k(["x", "a"], {"a"}, k=1) == 0
    assert precision_at_k([], {"a"}) == 0


def test_mrr():
    assert mrr(["a", "x"], {"a"}) == 1
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert mrr(["x"], {"a"}) == 0
    assert mrr([], {"a"}) == 0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x", "y", "z"], {"a", "b"}) == pytest.approx(1.0)


def test_ndcg_rewards_higher_ranks():
    assert ndcg_at_k(["a", "x"], {"a", "b"}) > ndcg_at_k(["x", "a"], {"a", "b"})


def test_ndcg_known_value():
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "y"], {"a", "b"}) == pytest.approx(expected)


def test_ndcg_only_counts_the_top_k():
    assert ndcg_at_k(["x", "y", "a"], {"a"}, k=2) == 0


def test_no_relevant_jobs_scores_zero():
    assert precision_at_k(["a"], set()) == 0
    assert mrr(["a"], set()) == 0
    assert ndcg_at_k(["a"], set()) == 0


def test_repeated_jobs_only_count_once():
    assert precision_at_k(["a", "a", "a", "a", "a"], {"a"}) == pytest.approx(1 / 5)
    assert ndcg_at_k(["a", "a"], {"a", "b"}) == ndcg_at_k(["a"], {"a", "b"})
    assert mrr(["x", "x", "a"], {"a"}) == pytest.approx(1 / 2)
