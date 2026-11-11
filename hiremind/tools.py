"""The four tools the Searcher can call. Plain Python functions."""
from hiremind.schemas import Job, Profile
from hiremind import config
from functools import lru_cache
import csv


@lru_cache(maxsize=1)
def _jobs() -> dict[str, Job]:
    with (config.PROCESSED_DIR / "jobs.csv").open(newline="", encoding="utf-8") as f:
        return {row["job_id"]: Job(
            job_id=row["job_id"], title=row["title"], description=row["description"],
            category=row["category"], skills=[s for s in row["skills"].split("|") if s],
            level=row["level"] or None,
        ) for row in csv.DictReader(f)}


@lru_cache(maxsize=1)
def _index():
    from hiremind import index
    return index.load()


def search_jobs(query: str, k: int = 20) -> list[Job]:
    """Semantic search over the FAISS job index."""
    return [get_job_details(job_id) for job_id, _ in _index().search(query, k)]


def filter_jobs(jobs: list[Job], category: str | None = None, level: str | None = None) -> list[Job]:
    return [job for job in jobs if (category is None or job.category == category) and (level is None or job.level == level)]


def get_job_details(job_id: str) -> Job:
    return _jobs()[str(job_id)]


def skill_overlap(profile: Profile, job: Job) -> float:
    """Share of the job's skills that are in the profile, between 0 and 1."""
    job_skills = {skill.casefold() for skill in job.skills}
    return len(job_skills & {skill.casefold() for skill in profile.skills}) / len(job_skills) if job_skills else 0.0


TOOLS = {
    "search_jobs": search_jobs,
    "filter_jobs": filter_jobs,
    "get_job_details": get_job_details,
    "skill_overlap": skill_overlap,
}
