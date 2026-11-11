"""Observable features for tuning relevance against the validation labels."""
from hiremind.schemas import Job, Profile
from hiremind.tools import skill_overlap


def relevance_features(profile: Profile, job: Job) -> dict:
    """Keep unknown experience and skills distinct from a known mismatch."""
    return {
        "same_category": profile.category == job.category,
        "skill_overlap": skill_overlap(profile, job) if job.skills else None,
        "resume_years_experience": profile.years_experience,
        "job_level": job.level or None,
    }
