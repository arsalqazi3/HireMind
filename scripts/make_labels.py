"""Make rule-based training labels for the Parser, Critic and Writer.

Run as: python -m scripts.make_labels

These labels come from rules, not real recruiters. They teach the model each
task but are not a perfect answer key.

Outputs JSONL files in data/processed/, one per role and split.
"""


def parser_labels(resume: dict) -> dict:
    """Category from the dataset label, skills and experience from fixed rules."""
    raise NotImplementedError


def critic_score(profile: dict, job: dict) -> tuple[float, str]:
    """Rule score out of 10 from category, skill overlap and level, with a short reason.
    Used for the about 20,000 resume and job pairs."""
    raise NotImplementedError


def writer_example(profile: dict, job: dict) -> str:
    """Fill a cover letter template using only facts from the profile and the job."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
