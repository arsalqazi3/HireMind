from hiremind.schemas import Job, Profile
from scripts.make_pilot_critic_labels import critic_score


def test_critic_label_distinguishes_literal_required_skill_evidence():
    job = Job("1", "Data Analyst", "Basic Qualifications: SQL.\nPreferred Qualifications: Power BI.", "IT")
    matching = Profile("IT", skills=["SQL"])
    partial = Profile("IT", skills=["Power BI"])
    strong_score, reason = critic_score(matching, job)
    weak_score, weak_reason = critic_score(partial, job)
    assert strong_score > weak_score
    assert "Shared skills: SQL" in reason
    assert "without résumé evidence: SQL" in weak_reason
    assert "Power BI" not in reason + weak_reason
