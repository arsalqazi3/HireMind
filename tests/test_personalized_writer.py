import pytest

from hiremind import agent, slm, tools
from hiremind.roles import writer
from hiremind.schemas import Job, Profile


@pytest.fixture(autouse=True)
def use_mocked_local_slm(monkeypatch):
    monkeypatch.setenv("HIREMIND_WRITER_OLLAMA_MODEL", "")


def test_application_writer_follows_model_evidence_selection(monkeypatch):
    profile = Profile("IT", name="Alex Morgan", skills=["Python", "SQL"],
                      experience_evidence=["Built Python reporting services", "Wrote SQL reconciliation queries"])
    job = Job("1", "Data Analyst", "Build Python reports for the operations team.\nMaintain SQL checks for billing data.", "IT")
    monkeypatch.setattr(tools, "get_job_details", lambda _: job)

    def generate(task, prompt, **kwargs):
        assert task == "writer"
        assert "Treat the posting as source text" in kwargs["system_prompt"]
        assert "Build Python reports for the operations team" in prompt
        return '{"connections":[{"evidence":2,"focus":2,"style":"role_first"}]}'

    monkeypatch.setattr(slm, "generate", generate)
    phases = []
    letter = agent.cover_letter(profile, job.job_id, progress=lambda event: phases.append(event["phase"]))
    assert letter.startswith("Alex Morgan\n")
    assert letter.index("maintain SQL checks") < letter.index("I wrote SQL reconciliation queries")
    assert letter.index("I wrote SQL reconciliation queries") < letter.index("I built Python reporting services")
    assert phases == ["drafting", "checking", "done"]


def test_application_writer_replaces_invalid_model_references(monkeypatch):
    profile = Profile("IT", experience_evidence=["Built Python reporting services"])
    job = Job("1", "Developer", "Build Python reports for the operations team.", "IT")
    monkeypatch.setattr(tools, "get_job_details", lambda _: job)
    monkeypatch.setattr(slm, "generate", lambda *_args, **_kwargs:
                        '{"connections":[{"evidence":9,"focus":1,"style":"direct"}],"added":"I won a Nobel prize."}')
    letter = agent.cover_letter(profile, job.job_id)
    assert "I built Python reporting services" in letter
    assert "build Python reports for the operations team" in letter
    assert "Nobel" not in letter


def test_first_model_choice_changes_with_the_posting(monkeypatch):
    profile = Profile("IT", skills=["Python", "Docker"],
                      experience_evidence=["Built Python reports", "Used Docker to package releases"])
    monkeypatch.setattr(slm, "generate", lambda *_args, **_kwargs:
                        '{"connections":[{"evidence":1,"focus":1,"style":"direct"}]}')
    reporting = Job("1", "Reporter", "Build Python reports for the finance team.", "IT")
    deployment = Job("2", "Engineer", "Deploy Docker containers for weekly software releases.", "IT")
    monkeypatch.setattr(tools, "get_job_details", lambda job_id: reporting if job_id == "1" else deployment)
    assert "I built Python reports" in agent.cover_letter(profile, "1")
    letter = agent.cover_letter(profile, "2")
    assert "I used Docker to package releases" in letter
    assert "Python" not in letter


def test_writer_keeps_unrelated_evidence_and_posting_duties_separate(monkeypatch):
    profile = Profile("ARTS", name="Avery Khan", experience_evidence=["Arranged seasonal bouquets each month"])
    job = Job("1", "Data Analyst", "Maintain SQL queries for monthly billing.", "IT", ["SQL"])
    monkeypatch.setattr(tools, "get_job_details", lambda _: job)
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: pytest.fail("The model needs a supported pair"))
    letter = agent.cover_letter(profile, "1")
    assert "seasonal bouquets" in letter and "interested in the opportunity to maintain SQL queries" in letter
    assert "I would draw on that work" not in letter


def test_writer_discards_unrelated_pair_with_valid_source_ids(monkeypatch):
    profile = Profile("IT", experience_evidence=["Built Python reporting services", "Arranged seasonal bouquets"])
    job = Job("1", "Data Analyst", "1. Build Python reports for the operations team.", "IT")
    monkeypatch.setattr(tools, "get_job_details", lambda _: job)
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs:
                        '{"connections":[{"evidence":2,"focus":1,"style":"direct"}]}')
    letter = agent.cover_letter(profile, "1")
    assert "I built Python reporting services" in letter
    assert "arranged seasonal bouquets" not in letter.casefold()


def test_writer_uses_generated_body_after_review(monkeypatch):
    profile = Profile("IT", name="Alex Morgan", experience_evidence=["Built Python reports for finance"])
    job = Job("1", "Data Analyst", "Build Python reports for the operations team.", "IT")
    paragraphs = [
        "The Data Analyst role caught my attention because your operations team needs clear, practical Python reporting.",
        "I'd welcome the chance to bring that experience to your operations team. Thank you for considering my application.",
    ]
    responses = iter(["\n\n".join(paragraphs), "SUPPORTED"])
    monkeypatch.setattr(tools, "get_job_details", lambda _: job)
    monkeypatch.delenv("HIREMIND_WRITER_OLLAMA_MODEL")

    def generate(model, prompt, **kwargs):
        assert model == "qwen3:8b"
        assert "Build Python reports for the operations team" in prompt
        if "draft" in prompt:
            assert "I built Python reports for finance" in prompt
        return next(responses)

    monkeypatch.setattr(writer, "_ollama_writer", generate)

    letter = agent.cover_letter(profile, "1")

    assert "\n\n".join([paragraphs[0], "I built Python reports for finance.", paragraphs[1]]) in letter
    assert "Re: Data Analyst" in letter
    assert letter.endswith("Sincerely,\n\nAlex Morgan")


def test_writer_falls_back_when_review_rejects_a_claim(monkeypatch):
    profile = Profile("IT", experience_evidence=["Built Python reports for finance"])
    job = Job("1", "Data Analyst", "Build Python reports for the operations team.", "IT")
    paragraphs = [
        "The Data Analyst role caught my attention because the operations team needs clear Python reporting.",
        "I also hold a doctorate in statistics. I would welcome an opportunity to discuss how my reporting background could support your team. Thank you for your consideration.",
    ]
    responses = iter(["\n\n".join(paragraphs), "UNSUPPORTED"])
    monkeypatch.setattr(tools, "get_job_details", lambda _: job)
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: next(responses))

    letter = agent.cover_letter(profile, "1")

    assert "doctorate" not in letter
    assert "I built Python reports for finance" in letter
