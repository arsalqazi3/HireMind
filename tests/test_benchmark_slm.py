import csv

from hiremind import config
from scripts.benchmark_slm import peak_rss_mib, validation_example


def test_peak_memory_is_reported_on_this_system():
    assert 1 < peak_rss_mib() < 100_000  # a running Python process uses more than 1 MiB


def test_example_uses_only_validation_resume(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    labels = tmp_path / "evaluation" / "labels"
    for directory in (processed, splits, labels):
        directory.mkdir(parents=True)
    (splits / "val.txt").write_text("2\n")

    def write(path, fields, rows):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write(labels / "relevance_pairs.csv", ["resume_id", "job_id"], [
        {"resume_id": "1", "job_id": "10"}, {"resume_id": "2", "job_id": "20"},
    ])
    write(processed / "resumes.csv", ["id", "text"], [{"id": "2", "text": "validation"}])
    write(processed / "jobs.csv", ["job_id", "title"], [{"job_id": "20", "title": "job"}])
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "SPLITS_DIR", splits)

    resume, job = validation_example()
    assert (resume["id"], job["job_id"]) == ("2", "20")
