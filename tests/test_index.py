"""Index tests with a fake embedder, so they run without downloading a model."""
import numpy as np
import pandas as pd
import pytest

from evaluation.baselines import faiss_only
from hiremind import index as job_index

VOCAB = ["cook", "nurse", "python", "teacher"]


def fake_embed(texts, show_progress=False):
    """One dimension per word, scaled to unit length like the real model."""
    vectors = np.array([[text.lower().count(word) for word in VOCAB] for text in texts], dtype="float32") + 1e-3
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def jobs_csv(tmp_path):
    path = tmp_path / "jobs.csv"
    pd.DataFrame({
        "job_id": [101, 102, 103],
        "title": ["Line Cook", "Registered Nurse", "Python Developer"],
        "skills": ["Food safety", None, "Python|SQL"],
        "description": ["cook meals", "nurse care", "python code"],
    }).to_csv(path, index=False)
    return path


def test_job_text_puts_title_and_skills_first():
    assert job_index.job_text("Cook", "Baking|Catering", "Long text") == "Cook\nBaking, Catering\nLong text"


def test_job_text_handles_jobs_with_no_skills():
    assert job_index.job_text("Cook", float("nan"), "Long text") == "Cook\n\nLong text"


def test_search_returns_the_closest_job_first(jobs_csv, tmp_path):
    built = job_index.build(jobs_csv, tmp_path / "index", embed_fn=fake_embed)
    results = built.search_vector(fake_embed(["python"])[0], k=2)
    assert results[0][0] == "103"
    assert results[0][1] > results[1][1]
    assert len(results) == 2


def test_saved_index_gives_the_same_results(jobs_csv, tmp_path):
    built = job_index.build(jobs_csv, tmp_path / "index", embed_fn=fake_embed)
    loaded = job_index.load(tmp_path / "index")
    query = fake_embed(["nurse"])[0]
    assert loaded.index.ntotal == 3
    assert loaded.search_vector(query, k=3) == built.search_vector(query, k=3)


def test_asking_for_more_jobs_than_exist_returns_what_there_is(jobs_csv, tmp_path):
    built = job_index.build(jobs_csv, tmp_path / "index", embed_fn=fake_embed)
    assert len(built.search_vector(fake_embed(["cook"])[0], k=50)) == 3


def test_faiss_only_returns_job_ids(jobs_csv, tmp_path, monkeypatch):
    built = job_index.build(jobs_csv, tmp_path / "index", embed_fn=fake_embed)
    monkeypatch.setattr(job_index, "embed", fake_embed)
    assert faiss_only("experienced cook", k=1, index=built) == ["101"]
