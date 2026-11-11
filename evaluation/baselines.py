"""Baselines that do not use the agent."""
from hiremind import config
from hiremind.index import JobIndex, load


def tfidf_logreg_category(train_texts: list[str], train_labels: list[str], test_texts: list[str]) -> list[str]:
    """TF-IDF plus Logistic Regression for the resume category."""
    raise NotImplementedError


def faiss_only(resume_text: str, k: int = config.TOP_K, index: JobIndex | None = None) -> list[str]:
    """Ablation A: embed the resume, return the k nearest job ids, no agent."""
    index = index or load()
    return [job_id for job_id, _ in index.search(resume_text, k)]
