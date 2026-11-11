"""FAISS index of the job pool, built with a small local embedding model."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from hiremind import config

INDEX_FILE = "jobs.faiss"
IDS_FILE = "job_ids.npy"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Return unit-length vectors, so the inner product is the cosine similarity."""
    vectors = _model().encode(
        texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=show_progress
    )
    return vectors.astype("float32")


def job_text(title: str, skills: str, description: str) -> str:
    """The text embedded for a job. Title and skills go first because the model
    only reads the start of a long text."""
    skills = "" if pd.isna(skills) else skills.replace("|", ", ")
    return f"{title}\n{skills}\n{description}"


@dataclass
class JobIndex:
    index: faiss.Index
    job_ids: np.ndarray

    def search_vector(self, vector: np.ndarray, k: int = config.TOP_K) -> list[tuple[str, float]]:
        """Return up to k (job_id, cosine similarity) pairs, best first."""
        scores, positions = self.index.search(vector.reshape(1, -1).astype("float32"), k)
        return [(str(self.job_ids[p]), float(s)) for p, s in zip(positions[0], scores[0]) if p != -1]

    def search(self, text: str, k: int = config.TOP_K) -> list[tuple[str, float]]:
        return self.search_vector(embed([text])[0], k)


def build(jobs_path: Path = config.PROCESSED_DIR / "jobs.csv", out_dir: Path = config.MODELS_DIR / "index", embed_fn=embed) -> JobIndex:
    """Embed every job and save the index and the id mapping."""
    jobs = pd.read_csv(jobs_path, usecols=["job_id", "title", "skills", "description"])
    texts = [job_text(t, s, d) for t, s, d in zip(jobs["title"], jobs["skills"], jobs["description"])]
    vectors = embed_fn(texts)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    np.save(out_dir / IDS_FILE, jobs["job_id"].to_numpy())
    return JobIndex(index, jobs["job_id"].to_numpy())


def load(index_dir: Path = config.MODELS_DIR / "index") -> JobIndex:
    index_dir = Path(index_dir)
    return JobIndex(faiss.read_index(str(index_dir / INDEX_FILE)), np.load(index_dir / IDS_FILE))
