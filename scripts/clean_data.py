"""Clean both Kaggle datasets.

Run as: python -m scripts.clean_data

Inputs (download by hand into data/raw/):
- resume-dataset (LiveCareer, 2,484 resumes, 24 categories)
- linkedin-job-postings (2023-2024, 124,000+ postings)

Outputs in data/processed/:
- resumes.csv: id, category, text
- jobs.csv: about config.JOB_POOL_SIZE postings, balanced across the 24
  categories using the rules in docs/category_mapping.md
"""


def clean_resumes(raw_dir: str) -> "pandas.DataFrame":
    """Drop empty and duplicate resumes, normalize whitespace."""
    raise NotImplementedError


def map_job_category(title: str, skills: list[str], industries: list[str]) -> str | None:
    """Map a posting to one of the 24 resume categories, or None if no rule matches."""
    raise NotImplementedError


def clean_jobs(raw_dir: str) -> "pandas.DataFrame":
    """Join postings with their skills and industries, map categories, dedupe, balance."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
