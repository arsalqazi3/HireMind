import pytest

from evaluation.review import agreement, combine, freeze_batch


def test_review_keeps_blanks_and_disagreements_out_of_consensus():
    a = {("r", "a"): "0", ("r", "b"): "1", ("r", "c"): "0"}
    b = {("r", "a"): "0", ("r", "b"): "", ("r", "c"): "1"}
    accepted, pending = combine(a, b)
    assert accepted == [("r", "a", "0"), ("r", "b", ""), ("r", "c", "")]
    assert [row[4] for row in pending] == ["missing judgment", "conflict"]
    accepted, pending = combine(a, b, {("r", "c"): "1"})
    assert accepted[-1] == ("r", "c", "1")
    assert len(pending) == 1
    with pytest.raises(ValueError, match="only pairs with conflicting"):
        combine(a, b, {("r", "b"): "1"})


def test_agreement_uses_only_pairs_with_two_votes():
    left = dict(enumerate(["0", "0", "1", "1", "1"]))
    right = dict(enumerate(["0", "1", "1", "1", ""]))
    assert agreement(left, right) == {
        "pairs": 5, "double_reviewed": 4, "agreed": 3, "disagreed": 1,
        "agreement": 0.75, "cohen_kappa": 0.5,
    }
    assert agreement({0: "0"}, {0: "0"})["cohen_kappa"] is None
    assert agreement({0: ""}, {0: "0"})["agreement"] is None


def test_freeze_batch_keeps_cohort_order_and_requires_every_selected_version():
    data = {"settings": {"split": "test", "resume_ids": ["2", "3", "1"]}, "runs": {
        "A": {"1": {"job_ids": ["a"]}, "2": {"job_ids": ["b"]}, "3": {"job_ids": ["c"]}},
        "E": {"1": {"job_ids": ["a"]}, "2": {"job_ids": ["d"]}},
    }}
    result = freeze_batch(data, 1, ["A", "E"])
    assert result["settings"] == {"split": "test", "resume_ids": ["2"]}
    assert list(result["runs"]["A"]) == list(result["runs"]["E"]) == ["2"]
    with pytest.raises(ValueError, match="E \\(1 pending\\)"):
        freeze_batch(data, 2, ["A", "E"])
