from time import monotonic

import pytest

from evaluation import ablation, calibrate
from scripts.run_ablation_study import collect_variant


def test_study_runner_enforces_deadline_before_starting_worker():
    assert collect_variant(["should-not-start"], "C", monotonic() - 1) == 124


def test_precision_uses_a_shared_fully_judged_cohort():
    data = {"settings": {"resume_ids": ["1", "2"]}, "runs": {
        "A": {"1": {"job_ids": ["a"], "seconds": 1},
              "2": {"job_ids": ["c"], "seconds": 1}},
        "D": {"1": {"job_ids": ["b"], "round_rankings": [["a"], ["b"]], "seconds": 2},
              "2": {"job_ids": ["d"], "seconds": 2}},
    }}
    judgments = {"1": {"a": False, "b": True}, "2": {"d": True}}
    report = ablation.score(data, judgments)
    assert report["A"]["completely_judged_resumes"] == report["D"]["completely_judged_resumes"] == 1
    assert report["A"]["precision_at_5"] == 0
    assert report["D"]["precision_at_5"] == 0.2
    assert report["D"]["retries_improving_precision"] == 1
    assert report["A"]["unjudged"] == 1


def test_judgments_refuse_unknown_or_duplicate_pairs(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("resume_id,job_id,relevant\nr,j,1\nr,j,0\n")
    with pytest.raises(ValueError, match="duplicate"):
        ablation.read_judgments(path, {"r": {"j"}})


def test_completed_empty_rankings_count_as_five_empty_slots():
    data = {"settings": {"resume_ids": ["r"]}, "runs": {"D": {"r": {"job_ids": [], "seconds": 1}}}}
    report = ablation.score(data, {"r": {}})["D"]
    assert report["completely_judged_resumes"] == 1
    assert report["precision_at_5"] == 0
    assert report["empty_slots"] == 5
    assert report["empty_result_resumes"] == 1
    assert report["five_result_resumes"] == 0


def test_checked_cohort_preserves_order_and_rejects_missing_labels(tmp_path):
    labels = tmp_path / "checked.csv"
    labels.write_text("resume_id,in_job_pool\n1,1\n2,0\n3,1\n")
    data = {"settings": {"split": "test", "resume_ids": ["3", "2", "1"]}, "runs": {
        "D": {resume_id: {"job_ids": []} for resume_id in ("1", "2", "3")},
    }}
    groups = ablation.split_checked_cohort(data, labels)
    assert groups["in_pool"]["settings"]["resume_ids"] == ["3", "1"]
    assert list(groups["in_pool"]["runs"]["D"]) == ["3", "1"]
    assert list(groups["out_of_pool"]["runs"]["D"]) == ["2"]
    labels.write_text("resume_id,in_job_pool\n1,1\n2,0\n")
    with pytest.raises(ValueError, match="missing test résumés"):
        ablation.split_checked_cohort(data, labels)


def test_review_pool_and_score_use_the_same_deduplicated_top_five():
    data = {"settings": {"resume_ids": ["r"]}, "runs": {
        "D": {"r": {"job_ids": ["a", "a", "b", "c", "d", "e"], "seconds": 1}},
    }}
    judgments = {"r": {job: job == "e" for job in "abcde"}}
    assert ablation.review_pool(data)["r"] == set("abcde")
    assert ablation.score(data, judgments)["D"]["precision_at_5"] == 0.2


def test_diagnostics_compare_only_shared_successful_cases_and_show_pending_variants():
    data = {"settings": {"split": "test", "resume_ids": ["1", "2", "3"]}, "runs": {
        "A": {"1": {"job_ids": ["a"], "seconds": 1, "category_matches": 1},
              "2": {"job_ids": ["x"], "seconds": 2, "category_matches": 1}},
        "D": {"1": {"job_ids": ["b"], "seconds": 4, "category_matches": 0, "retries": 1},
              "2": {"error": "ValueError", "job_ids": [], "seconds": 9},
              "3": {"job_ids": ["z"], "seconds": 6, "category_matches": 1}},
    }}
    result = ablation.diagnostics(data)
    assert result["planned_cases"] == 15
    assert result["variants"]["E"]["pending"] == 3
    assert result["variants"]["D"]["errors"] == 1
    assert result["variants"]["D"]["p90_seconds"] == 6
    assert result["comparisons"]["A–D"]["shared_successful_resumes"] == 1
    assert result["comparisons"]["A–D"]["same_category_slots_delta"] == -1


def test_paired_precision_bootstrap_uses_only_fully_judged_resumes():
    data = {"settings": {"resume_ids": ["1", "2", "3"]}, "runs": {
        "A": {resume: {"job_ids": ["a"]} for resume in ("1", "2", "3")},
        "D": {resume: {"job_ids": ["d"]} for resume in ("1", "2", "3")},
    }}
    judgments = {"1": {"a": False, "d": True}, "2": {"a": False, "d": True}, "3": {"a": True}}
    result = ablation.compare_precision(data, judgments, "A")
    assert result["fully_judged_resumes"] == 2
    assert result["baseline_precision_at_5"] == 0
    assert result["comparisons"]["D"] == {
        "precision_at_5": 0.2, "delta_precision_at_5": 0.2, "bootstrap_95_interval": [0.2, 0.2],
        "improved": 2, "unchanged": 0, "worse": 0,
    }


def test_calibration_requires_both_review_classes_and_multiple_resumes():
    with pytest.raises(ValueError, match="five reviewed pairs"):
        calibrate.fit([("r", {"same_category": 1.0}, True)] * 12)
    examples = [(str(i), {"same_category": float(i % 2)}, bool(i % 2)) for i in range(10)]
    result = calibrate.fit(examples)
    assert result["out_of_fold_pairs"] == 10
    assert result["coefficients"]["same_category"] > 0
