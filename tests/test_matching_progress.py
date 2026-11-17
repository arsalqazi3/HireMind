from hiremind import agent, matching
import pytest
from hiremind.roles import critic, parser, searcher
from hiremind.schemas import Job, Judgement, Profile


def test_evidence_adjustment_breaks_ties_and_discourages_network_mismatch():
    profile = Profile("INFORMATION-TECHNOLOGY", skills=["Python", "SQL"], roles=["Software Engineer"])
    software = Job("1", "Software Engineer", "Requirements: Python and SQL.", "INFORMATION-TECHNOLOGY", ["Python", "SQL"])
    network = Job("2", "Network Engineer", "Requirements: Cisco and Python.", "INFORMATION-TECHNOLOGY", ["Python"])
    good = matching.adjust(profile, software, Judgement("1", 7, "Invented claim"))
    poor = matching.adjust(profile, network, Judgement("2", 7, "Invented claim"))
    assert good.score > poor.score
    assert poor.score < 6
    assert "Invented claim" not in good.reason
    assert "Python" in good.reason


def test_senior_leadership_is_not_ranked_high_for_shorter_non_leadership_profile():
    job = Job("1", "Principal Software Engineer", "Requirements: Python", "INFORMATION-TECHNOLOGY", ["Python"])
    junior = Profile("INFORMATION-TECHNOLOGY", skills=["Python"], years_experience=4, roles=["Software Engineer"])
    senior = Profile("INFORMATION-TECHNOLOGY", skills=["Python"], years_experience=12, roles=["Engineering Manager"])
    assert matching.adjust(junior, job, Judgement("1", 8, "ok")).score < 6
    assert matching.adjust(senior, job, Judgement("1", 8, "ok")).score >= 6


def test_progress_reports_actual_candidate_counts_and_estimate(monkeypatch):
    jobs = [Job(str(i), "Developer", "", "IT") for i in range(3)]
    monkeypatch.setattr(parser, "parse", lambda text: Profile("IT"))
    monkeypatch.setattr(searcher, "search", lambda memory: jobs)
    monkeypatch.setattr(critic, "judge", lambda profile, jobs: [Judgement(jobs[0].job_id, 7, "ok")])
    events = []
    result = agent.run("resume", progress=events.append)
    scores = [event for event in events if event["phase"] == "scoring"]
    assert [event["completed"] for event in scores] == [0, 1, 2, 3]
    assert scores[1]["remaining_seconds"] is None
    assert scores[2]["remaining_seconds"] >= 0
    assert events[-1]["phase"] == "done"
    assert len(result.top_jobs) == 3


def test_parser_recovers_explicit_years_and_software_category(monkeypatch):
    from hiremind import slm
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"category":"ENGINEERING","roles":["Software Engineer"]}')
    profile = parser.parse("Software Engineer with 4 years of experience in Python.")
    assert profile.category == "INFORMATION-TECHNOLOGY"
    assert profile.years_experience == 4


def test_ranking_handles_different_occupation_alternatives_and_experience():
    profile = Profile("INFORMATION-TECHNOLOGY", skills=["Python", "JavaScript", "REST APIs"],
                      roles=["Software Engineer"], years_experience=4)
    jobs = [
        Job("trading", "Senior Software Engineer", "Requirements: At least 2 years experience as a software developer. Strong programming skills in either Python, JavaScript or C++.", "INFORMATION-TECHNOLOGY", ["Python", "JavaScript", "C++"]),
        Job("api", "Senior API Developer", "Requirements: 10 years of experience developing REST APIs.", "INFORMATION-TECHNOLOGY", ["REST APIs"]),
        Job("design", "Junior Graphic Designer", "Graphic design and Adobe Illustrator.", "DESIGNER", []),
    ]
    judgements = [matching.adjust(profile, job, Judgement(job.job_id, 7, "")) for job in jobs]
    assert judgements[0].score > judgements[1].score > judgements[2].score
    assert "C++" not in judgements[0].reason
    assert "10 years" in judgements[1].reason
    assert "no shared skills" in judgements[2].reason.lower()


def test_director_typo_still_flags_senior_role():
    profile = Profile("FINANCE", years_experience=3, roles=["Data Analyst"])
    job = Job("1", "Finance Directore", "", "FINANCE")
    assert matching.adjust(profile, job, Judgement("1", 8, "")).score < 6


@pytest.mark.parametrize(("text", "expected"), [
    ("At least three years of industry experience", 3),
    ("Minimum of 8 to 10 years of recent, active development experience", 8),
    ("5 years’ experience in accounting", 5),
    ("At least (5) five years’ experience in an office setting", 5),
    ("5+ years experience in development", 5),
    ("5+ years experienceBe comfortable in a leadership role", 5),
    ("A bachelor's degree or equivalent", None),
])
def test_experience_requirement_formats(text, expected):
    assert matching._experience_years(text) == expected


def test_early_career_unknown_experience_is_not_assumed_zero():
    p = Profile("IT", early_career=True)
    j = Job("1", "Developer", "At least three years of industry experience", "IT")
    result = matching.adjust(p, j, Judgement("1", 7, ""))
    assert result.score < 7
    assert "not established" in result.reason
