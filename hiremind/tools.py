"""The four tools the Searcher can call. Plain Python functions."""
from hiremind.schemas import Job, Profile


def search_jobs(query: str, k: int = 20) -> list[Job]:
    """Semantic search over the FAISS job index."""
    raise NotImplementedError


def filter_jobs(jobs: list[Job], category: str | None = None, level: str | None = None) -> list[Job]:
    raise NotImplementedError


def get_job_details(job_id: str) -> Job:
    raise NotImplementedError


def skill_overlap(profile: Profile, job: Job) -> float:
    """Share of the job's skills that are in the profile, between 0 and 1."""
    raise NotImplementedError


TOOLS = {
    "search_jobs": search_jobs,
    "filter_jobs": filter_jobs,
    "get_job_details": get_job_details,
    "skill_overlap": skill_overlap,
}
