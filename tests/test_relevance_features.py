from evaluation.features import relevance_features
from hiremind.schemas import Job, Profile


def test_relevance_features_distinguish_unknown_from_zero():
    profile = Profile("IT", skills=["Python"], years_experience=None)
    known = Job("1", "Analyst", "", "IT", skills=["SQL"], level="Entry level")
    unknown = Job("2", "Analyst", "", "IT")
    assert relevance_features(profile, known) == {
        "same_category": True, "skill_overlap": 0.0,
        "resume_years_experience": None, "job_level": "Entry level",
    }
    assert relevance_features(profile, unknown)["skill_overlap"] is None
    assert relevance_features(profile, unknown)["job_level"] is None
