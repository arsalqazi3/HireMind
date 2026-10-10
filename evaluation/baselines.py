"""Baselines that do not use the agent."""


def tfidf_logreg_category(train_texts: list[str], train_labels: list[str], test_texts: list[str]) -> list[str]:
    """TF-IDF plus Logistic Regression for the resume category."""
    raise NotImplementedError


def faiss_only(resume_text: str, k: int = 5) -> list[str]:
    """Ablation A: embed the resume, return the k nearest job ids, no agent."""
    raise NotImplementedError
