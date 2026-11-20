import sqlite3
from contextlib import closing

from evaluation.hybrid_retrieval import build_lexical_index, compare, fuse, lexical_search
from hiremind.schemas import Job


def test_fusion_rewards_agreement_and_counts_duplicate_once():
    assert fuse(["dense", "shared", "shared"], ["lexical", "shared"]) == ["shared", "dense", "lexical"]


def test_bm25_treats_query_punctuation_as_text():
    with closing(sqlite3.connect(":memory:")) as connection:
        build_lexical_index(connection, [Job("1", "Python engineer", "Build pipelines", "IT")])
        assert lexical_search(connection, 'Python " OR )', 5) == ["1"]
        assert lexical_search(connection, "the and 123", 5) == []


def test_comparison_preserves_unjudged_jobs_for_human_review():
    jobs = {
        "1": Job("1", "Accountant", "Audit financial statements", "ACCOUNTANT"),
        "2": Job("2", "Engineer", "Build pipelines", "IT"),
    }

    class DenseIndex:
        def search(self, query, depth):
            return [("2", 0.7), ("1", 0.6)]

    result = compare([{"id": "r", "text": "Accountant", "category": "ACCOUNTANT"}],
                     jobs, DenseIndex(), {"r": {"1": True}}, depth=20)
    assert result["runs"]["hybrid"]["r"]["job_ids"] == ["1", "2"]
    report = result["summary"]["hybrid"]["at_5"]
    assert report["confirmed_relevant"] == 1
    assert report["unjudged"] == 1
    assert report["precision_bounds"] == [0.2, 0.4]
