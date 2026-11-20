"""The four tools the Searcher can call. Plain Python functions."""
from hiremind.schemas import Job, Profile
from hiremind import config, matching
from functools import lru_cache
import csv
import re


@lru_cache(maxsize=1)
def _jobs() -> dict[str, Job]:
    with (config.PROCESSED_DIR / "jobs.csv").open(newline="", encoding="utf-8") as f:
        return {row["job_id"]: Job(
            job_id=row["job_id"], title=row["title"], description=row["description"],
            category=row["category"], skills=[s for s in row["skills"].split("|") if s],
            level=row["level"] or None,
            location=row.get("location") or None, work_type=row.get("work_type") or None,
        ) for row in csv.DictReader(f)}


@lru_cache(maxsize=1)
def _index():
    from hiremind import index
    return index.load()


def search_jobs(query: str, k: int = 20) -> list[Job]:
    """Semantic search over the FAISS job index."""
    return [get_job_details(job_id) for job_id, _ in _index().search(query, k)]


def _title_terms(title: str) -> set[str]:
    words = set(re.findall(r"[^\W\d_]+", title.casefold()))
    return words - {"a", "an", "and", "in", "of", "the", "junior", "jr", "senior", "sr"}


def shortlist_jobs(query: str, profile: Profile, exclude: set[str] | None = None) -> list[Job]:
    """Rerank a wider semantic pool using resume evidence before model review."""
    exclude = exclude or set()
    indexed = _index().search(query, config.RETRIEVAL_POOL + len(exclude))
    roles = [terms for role in profile.roles[:4] if (terms := _title_terms(role))]
    skills = {skill.casefold() for skill in profile.skills}
    jobs = _jobs()
    candidates = []
    considered = 0
    for job_id, similarity in indexed:
        if job_id in exclude:
            continue
        if considered == config.RETRIEVAL_POOL:
            break
        considered += 1
        job = jobs[job_id]
        if matching.early_seniority_conflict(profile, job) or matching.missing_title_credentials(profile, job):
            continue
        title = _title_terms(job.title)
        role_match = max((len(role & title) / len(role | title) for role in roles), default=0.0)
        shared = len(skills.intersection(skill.casefold() for skill in job.skills))
        score = similarity + 0.12 * role_match + 0.02 * min(shared, 4)
        if job.category == profile.category:
            score += 0.025
        candidates.append((score, job))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return [job for _, job in candidates[:config.REVIEW_LIMIT]]


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
