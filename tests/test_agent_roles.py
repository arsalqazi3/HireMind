import json

import pytest

from hiremind import agent, slm
from hiremind.roles import critic, parser, searcher, writer
from hiremind.schemas import Job, Judgement, Profile


def test_parser_discards_unbacked_facts_and_retries(monkeypatch):
    answers = iter(["invalid", json.dumps({"category": "information technology", "years_experience": 10,
        "education": ["Degree in Mathematics", "Imaginary PhD"], "roles": ["Data Analyst"], "projects": []})])
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: next(answers))
    profile = parser.parse("Data Analyst. Degree in Mathematics. Python and SQL.")
    assert profile.category == "INFORMATION-TECHNOLOGY"
    assert profile.years_experience is None
    assert profile.education == ["Degree in Mathematics"]
    assert profile.roles == ["Data Analyst"]
    assert "Python" in profile.skills


def test_critic_rejects_invalid_scores(monkeypatch):
    responses = iter(['{"score":8,"reason":"Relevant skills"}', '{"score":100,"reason":"invalid"}', 'invalid'])
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: next(responses))
    jobs = [Job(str(i), "Job", "Description", "IT") for i in range(3)]
    assert critic.judge(Profile("IT"), jobs) == [Judgement("0", 8, "Relevant skills")]
    assert critic.feedback(Profile("IT"), []) is not None


def test_writer_never_returns_unverified_draft(monkeypatch):
    monkeypatch.setattr(writer, "write", lambda *args: "I have a PhD and ten years of experience.")
    profile = Profile("IT", skills=["Python"])
    letter = writer.write_checked(profile, Job("1", "Developer", "", "IT"))
    assert "PhD" not in letter
    assert "Python" in letter
    assert writer.unsupported_claims(letter, profile) == []
    assert writer.unsupported_claims("I have a PhD.", profile) == ["I have a PhD."]


def test_agent_retries_and_returns_only_acceptable_ranked_jobs(monkeypatch):
    profile = Profile("IT")
    monkeypatch.setattr(parser, "parse", lambda _: profile)
    calls = []
    def fake_search(memory):
        calls.append(list(memory.episodic.critic_feedback))
        return [Job(str(len(calls)), "Role", "", "IT")]
    monkeypatch.setattr(searcher, "search", fake_search)
    monkeypatch.setattr(critic, "judge", lambda profile, jobs: [Judgement(jobs[0].job_id, 4 if len(calls) == 1 else 8, "reason")])
    result = agent.run("resume")
    assert result.retries == 1
    assert [j.job_id for j in result.top_jobs] == ["2"]
    assert calls[1]


def test_agent_without_critic(monkeypatch):
    monkeypatch.setattr(parser, "parse", lambda _: Profile("IT"))
    monkeypatch.setattr(searcher, "search", lambda _: [Job("1", "Job", "", "IT")])
    monkeypatch.setattr(critic, "judge", lambda *args: pytest.fail("critic should not run"))
    result = agent.run("resume", use_critic=False, use_episodic=False)
    assert result.top_jobs[0].job_id == "1"
    assert result.retries == 0
