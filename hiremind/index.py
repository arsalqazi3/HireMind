"""FAISS index of the job pool, built with a small local embedding model."""


def build(jobs_path: str, out_dir: str) -> None:
    """Embed every job and save the index and the id mapping."""
    raise NotImplementedError


def load(index_dir: str):
    raise NotImplementedError
