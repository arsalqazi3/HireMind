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


def test_parser_retries_when_category_is_not_supported(monkeypatch):
    answers = iter(['{"category":"SENIOR ACCOUNTANT"}', '{"category":"ACCOUNTANT"}'])
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: next(answers))
    assert parser.parse("Accountant with bookkeeping experience.").category == "ACCOUNTANT"


def test_parser_keeps_literal_skills_after_malformed_output(monkeypatch):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: "invalid")
    profile = parser.parse("Used Python and SQL to prepare reports.")
    assert profile.category == "UNKNOWN"
    assert {"Python", "SQL"} <= set(profile.skills)
    assert "unknown" not in searcher.profile_queries(profile)[0].casefold()
    with pytest.raises(ValueError, match="enough résumé information"):
        parser.parse("Hello")


def test_critic_rejects_invalid_scores(monkeypatch):
    responses = iter(['{"score":8,"reason":"Relevant skills"}', '{"score":100,"reason":"invalid"}', 'invalid'])
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: next(responses))
    jobs = [Job(str(i), "Job", "Description", "IT") for i in range(3)]
    assert critic.judge(Profile("IT"), jobs) == [Judgement("0", 8, "Relevant skills")]
    assert critic.feedback(Profile("IT"), []) is not None


def test_writer_never_returns_unverified_draft(monkeypatch):
    profile = Profile("IT", skills=["Python"])
    job = Job("1", "Developer", "", "IT")
    letter = writer.write_checked(profile, job)
    assert "PhD" not in letter
    assert "Python" in letter
    assert "Developer position" in letter
    assert writer.unsupported_claims(letter, profile, job) == []
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
    assert result.retries == 2
    assert [j.job_id for j in result.top_jobs] == ["2", "3"]
    assert calls[1]


def test_agent_without_critic(monkeypatch):
    monkeypatch.setattr(parser, "parse", lambda _: Profile("IT"))
    monkeypatch.setattr(searcher, "search", lambda _: [Job("1", "Job", "", "IT")])
    monkeypatch.setattr(critic, "judge", lambda *args: pytest.fail("critic should not run"))
    result = agent.run("resume", use_critic=False, use_episodic=False)
    assert result.top_jobs[0].job_id == "1"
    assert result.retries == 0


def test_parser_recovers_literal_sections_when_model_returns_placeholders(monkeypatch):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"category":"ENGINEERING","roles":["null"],"education":["null"],"projects":["null"]}')
    text = "PROFILE\nSoftware engineer with 4 years of experience.\nEXPERIENCE\nSoftware Engineer, Example Inc | 2022–2026\n- Built software, with Python.\nEDUCATION\nComputer Science Degree, Example University\nPROJECTS\nBuilt a Python tool."
    profile = parser.parse(text)
    assert profile.roles == ["Software Engineer"]
    assert profile.education == ["Computer Science Degree, Example University"]
    assert profile.projects == ["Built a Python tool."]
    assert profile.category == "INFORMATION-TECHNOLOGY"


def test_parser_recovers_alias_headings_and_preserves_wrapped_evidence(monkeypatch):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"category":"ENGINEERING"}')
    text = ("Alex Morgan\nProfessional Experience:\nReporting Analyst, Harbor Data\n"
            "- Built a finance reporting dashboard\nand saved two hours each week.\n"
            "Technical Skills\nPython and SQL\nCertifications\nData Analysis\n"
            "Selected Projects\nDesigned an invoice dashboard\nand shared it with the team.\n"
            "Volunteering\nOrganized community workshops.\n")
    profile = parser.parse(text)
    assert profile.roles == ["Reporting Analyst"]
    assert profile.experience_evidence == ["Built a finance reporting dashboard and saved two hours each week."]
    assert profile.projects == ["Designed an invoice dashboard and shared it with the team."]


def test_personalized_letter_uses_resume_name_and_relevant_skills():
    profile = Profile("INFORMATION-TECHNOLOGY", name="Alex Morgan", skills=["Python", "JavaScript", "Docker"],
                      years_experience=4, roles=["Software Engineer"], projects=["Built a Python reporting tool."])
    job = Job("1", "Backend Engineer", "", "INFORMATION-TECHNOLOGY", ["Python", "JavaScript"])
    letter = writer.write_checked(profile, job)
    assert letter.startswith("Alex Morgan\n")
    assert letter.endswith("Alex Morgan")
    assert "Python, JavaScript" in letter and "Docker" not in letter
    assert "I built a Python reporting tool" in letter
    assert "[Your name]" not in letter
    assert writer.unsupported_claims(letter, profile, job) == []


def test_name_extraction_ignores_job_titles_and_allows_unknown():
    assert parser.extract_name("Alex Morgan\nSoftware Engineer\n") == "Alex Morgan"
    assert parser.extract_name("TEST RESUME — FICTIONAL\nSamira Khan\nData Analyst\n") == "Samira Khan"
    assert parser.extract_name("Software Engineer\nPROFILE\nPython, SQL\n") is None


def test_cover_letter_uses_resume_experience_evidence_without_inventing():
    p = Profile("IT", name="Taylor Ahmed", skills=["HTML", "JavaScript"], roles=["Software Development Intern"],
                experience_evidence=["Updated website pages using HTML and JavaScript.", "Worked with a developer to test fixes."])
    job = Job("1", "Web Developer", "Build websites with HTML and JavaScript.", "IT", ["HTML", "JavaScript"])
    letter = writer.write_checked(p, job)
    assert "I updated website pages using HTML and JavaScript." in letter
    assert "I worked with a developer to test fixes." in letter
    assert writer.unsupported_claims(letter, p, job) == []


def test_recent_graduate_flags_senior_title_without_inventing_years(monkeypatch):
    from hiremind import matching
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"category":"INFORMATION-TECHNOLOGY"}')
    p = parser.parse("Taylor Ahmed\nPROFILE\nRecent computer science graduate seeking a junior web developer role.\nSKILLS\nPython\nEXPERIENCE\nSoftware Development Intern, Example Studio | Summer 2025\n- Updated website pages using Python.")
    assert p.early_career and p.years_experience is None
    assert p.experience_evidence == ["Updated website pages using Python."]
    senior = Job("s", "Senior Cloud Architect", "Seeking an experienced cloud architect.", "INFORMATION-TECHNOLOGY", ["Python"])
    misleading = Job("j", "Senior Software Engineer", "We are looking for talented Junior Software Engineers.", "INFORMATION-TECHNOLOGY", ["Python"])
    assert matching.adjust(p, senior, Judgement("s", 8, "")).score < 6
    assert matching.adjust(p, misleading, Judgement("j", 8, "")).score >= 6


def test_employer_text_is_not_promoted_to_a_project(monkeypatch):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"category":"ACCOUNTANT","projects":["Example Manufacturing | 2022–2026"]}')
    p = parser.parse("Avery Reed\nEXPERIENCE\nStaff Accountant, Example Manufacturing | 2022–2026\n- Prepared financial reports.\nEDUCATION\nAccounting degree")
    assert not p.projects
    assert "project" not in writer.write_checked(p, Job("1", "Accountant", "", "ACCOUNTANT")).lower()


def test_clear_occupation_corrects_unrelated_model_category(monkeypatch):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"category":"FINANCE"}')
    accountant = parser.parse("Avery Reed\nEXPERIENCE\nStaff Accountant, Example Manufacturing | 2022–2026")
    analyst = parser.parse("Samira Khan\nEXPERIENCE\nData Analyst, Example Retail | 2023–2026")
    assert accountant.category == "ACCOUNTANT"
    assert analyst.category == "INFORMATION-TECHNOLOGY"
