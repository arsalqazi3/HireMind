import csv

import pytest

from hiremind import config
from scripts.validate_parser import CATEGORIES, checked_test_resumes, validation_resumes


def test_validation_sample_excludes_other_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPLITS_DIR", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    (tmp_path / "val.txt").write_text("\n".join(str(i) for i in range(5)))
    with (tmp_path / "resumes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "text"])
        writer.writeheader()
        writer.writerow({"id": "99", "category": CATEGORIES[0], "text": "not validation"})
        for i, category in enumerate(CATEGORIES):
            writer.writerow({"id": str(i), "category": category, "text": "validation"})
    assert [row["id"] for row in validation_resumes()] == [str(i) for i in range(5)]
    (tmp_path / "val.txt").write_text("0\n")
    with pytest.raises(ValueError, match="Missing validation categories"):
        validation_resumes()
    with pytest.raises(ValueError, match="Missing validation categories"):
        validation_resumes(offset=1)


def test_checked_test_resumes_uses_both_references_and_excludes_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPLITS_DIR", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    (tmp_path / "test.txt").write_text("1\n2\n")
    (tmp_path / "resumes.csv").write_text("id,category,text\n1,ARTS,text\n2,ARTS,other text\n")
    labels = tmp_path / "checked.csv"
    labels.write_text("resume_id,dataset_category,checked_category,in_job_pool\n1,ARTS,TEACHER,1\n2,ARTS,NONE,0\n")
    assert checked_test_resumes(labels) == [
        {"id": "1", "category": "ARTS", "checked_category": "TEACHER", "text": "text"},
    ]
    (tmp_path / "test.txt").write_text("1\n")
    with pytest.raises(ValueError, match="unique test résumé IDs"):
        checked_test_resumes(labels)
