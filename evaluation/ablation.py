"""Matching evaluation and ablation study.

Runs the five versions on the same about 100 unseen test resumes, the same
job pool and the same search budget, and reports Precision@5, MRR, NDCG@5,
how often a Critic retry improved the ranking, and run times.

A. FAISS search only               baseline with no agent
B. Agent without the Critic        what self-correction adds
C. Agent without episodic memory   what memory adds
D. Full agent with the base SLM    what fine-tuning adds
E. Full agent with our fine-tuned SLM

Results go to reports/.
"""

VERSIONS = {
    "A": {"agent": False},
    "B": {"agent": True, "use_critic": False, "use_episodic": True, "model": "finetuned"},
    "C": {"agent": True, "use_critic": True, "use_episodic": False, "model": "finetuned"},
    "D": {"agent": True, "use_critic": True, "use_episodic": True, "model": "base"},
    "E": {"agent": True, "use_critic": True, "use_episodic": True, "model": "finetuned"},
}


def is_relevant(profile: dict, job: dict) -> bool:
    """Relevant on category, skills and level, with cut-offs set on validation data."""
    raise NotImplementedError


def run_version(name: str, resumes: list[dict]) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
