import pandas as pd
import pytest

from scripts.import_relevance_labels import agreement, problems, read_labels


def write_sheet(path, values, notes=None):
    pd.DataFrame({
        "pair_id": [f"P{i:03d}" for i in range(1, len(values) + 1)],
        "resume": "r", "job_title": "t", "company": "c", "job_level": "", "job_description": "d",
        "relevant": values,
        "note": notes or [None] * len(values),
    }).to_excel(path, sheet_name="labels", index=False)


def test_reads_numbers_and_text_the_same_way(tmp_path):
    path = tmp_path / "a.xlsx"
    write_sheet(path, [1, 0, "1", " 0 "])
    labels = read_labels("a", path)
    assert labels["relevant"].tolist() == ["1", "0", "1", "0"]
    assert labels["labeller"].unique().tolist() == ["a"]
    assert problems(labels) == []


def test_reports_empty_and_invalid_rows(tmp_path):
    path = tmp_path / "a.xlsx"
    write_sheet(path, [1, None, "yes"])
    assert problems(read_labels("a", path)) == ["a P002: empty", "a P003: yes"]


def test_agreement_on_shared_pairs():
    labels = pd.DataFrame({
        "pair_id": ["P001", "P002", "P003", "P004", "P001", "P002", "P003", "P009"],
        "labeller": ["a"] * 4 + ["b"] * 4,
        "relevant": ["1", "0", "1", "0", "1", "0", "0", "1"],
    })
    result = agreement(labels)
    assert result["pairs"] == 3
    assert result["same_answer"] == pytest.approx(2 / 3)
    assert -1 <= result["cohen_kappa"] <= 1
