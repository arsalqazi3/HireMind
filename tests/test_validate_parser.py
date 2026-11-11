import csv

import pytest

from hiremind import config
from scripts.validate_parser import CATEGORIES, validation_resumes


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
