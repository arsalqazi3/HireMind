import json

import pytest

from hiremind import config, slm, tools
from hiremind.memory import Memory
from hiremind.roles import searcher
from hiremind.schemas import Job, Profile


def test_search_executes_action_and_stops(monkeypatch):
    job = Job("1", "Accountant", "Work", "ACCOUNTANT", ["Excel"])
    responses = iter([
        '```json\n{"tool":"search_jobs","arguments":{"query":"accountant excel"}}\n```',
        '{"tool":"stop","arguments":{}}',
    ])
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(tools, "shortlist_jobs", lambda *args: [job])
    memory = Memory(semantic=Profile("ACCOUNTANT", skills=["Excel"]))
    assert searcher.search(memory) == [job]
    assert memory.episodic.queries == ["accountant excel"]
    assert memory.episodic.seen_job_ids == {"1"}
    assert [step.action for step in memory.short_term] == ["search_jobs", "stop"]


@pytest.mark.parametrize("response", [
    '{"tool":"search_jobs","arguments":"accountant"}',
    '{"tool":"unknown","arguments":{}}',
    '{"tool":"search_jobs","arguments":{"query":"one two three four five six seven eight nine"}}',
    '{"tool":"search_jobs","arguments":{"query":"job and skills"}}',
    'not json',
])
def test_invalid_actions_are_not_executed(monkeypatch, response):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: response)
    monkeypatch.setattr(tools, "shortlist_jobs", lambda *args: pytest.fail("invalid action executed"))
    memory = Memory(semantic=Profile("ACCOUNTANT"))
    assert searcher.search(memory) == []
    assert memory.short_term[-1].action == "error"


def test_repeated_query_is_not_executed(monkeypatch):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: json.dumps({"tool": "search_jobs", "arguments": {"query": "ACCOUNTANT"}}))
    memory = Memory(semantic=Profile("ACCOUNTANT"))
    memory.episodic.queries.append("accountant")
    with pytest.raises(ValueError, match="Repeated"):
        searcher.next_action(memory)


def test_repeated_query_stops_with_existing_candidates(monkeypatch):
    job = Job("1", "Accountant", "Work", "ACCOUNTANT")
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"tool":"search_jobs","arguments":{"query":"accountant"}}')
    monkeypatch.setattr(tools, "shortlist_jobs", lambda *args: [job])
    memory = Memory(semantic=Profile("ACCOUNTANT"))
    assert searcher.search(memory) == [job]
    assert [step.action for step in memory.short_term] == ["search_jobs", "stop"]
    assert memory.short_term[-1].feedback == "Repeated search query"


def test_skill_overlap_and_filter():
    job = Job("1", "Accountant", "", "ACCOUNTANT", ["Excel", "SQL"], "Entry level")
    assert tools.skill_overlap(Profile("ACCOUNTANT", skills=["excel"]), job) == 0.5
    assert tools.skill_overlap(Profile("ACCOUNTANT"), Job("2", "Other", "", "OTHER")) == 0
    assert tools.filter_jobs([job], level="Entry level") == [job]
    assert tools.filter_jobs([job], category="OTHER") == []


def test_retry_uses_alternative_query_after_repeat(monkeypatch):
    job = Job("2", "Developer", "", "INFORMATION-TECHNOLOGY")
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"tool":"search_jobs","arguments":{"query":"developer"}}')
    queries = []
    monkeypatch.setattr(tools, "shortlist_jobs", lambda query, *_: queries.append(query) or [job])
    memory = Memory(semantic=Profile("INFORMATION-TECHNOLOGY", skills=["Python"], roles=["Software Engineer"]))
    memory.episodic.queries.append("developer")
    assert searcher.search(memory) == [job]
    assert queries == ["Software Engineer Python"]
    assert "alternative query" in memory.short_term[-1].feedback


def test_applicant_name_alone_is_not_used_as_job_query(monkeypatch):
    monkeypatch.setattr(slm, "generate", lambda *args, **kwargs: '{"tool":"search_jobs","arguments":{"query":"Taylor Ahmed"}}')
    memory = Memory(semantic=Profile("IT", name="Taylor Ahmed"))
    with pytest.raises(ValueError, match="applicant name"):
        searcher.next_action(memory)


def test_shortlist_promotes_evidenced_jobs_beyond_dense_cutoff(monkeypatch):
    matches = [Job(str(i), "Team Lead", "", "OTHER") for i in range(20)]
    wanted = Job("fit", "Data Analyst", "", "IT", ["SQL"])
    seen = Job("seen", "Data Analyst", "", "IT", ["SQL"])
    senior = Job("senior", "Senior Data Analyst", "", "IT", ["SQL"])
    specialist = Job("specialist", "WalkMe Data Analyst", "", "IT", ["SQL"])

    class Index:
        def search(self, query, k):
            return ([("seen", 0.8), ("senior", 0.75), ("specialist", 0.7)]
                    + [(job.job_id, 0.60) for job in matches] + [("fit", 0.55)])[:k]

    monkeypatch.setattr(tools, "_index", Index)
    monkeypatch.setattr(tools, "_jobs", lambda: {job.job_id: job for job in [*matches, wanted, seen, senior, specialist]})
    profile = Profile("IT", skills=["sql"], roles=["Data Analyst"], early_career=True)
    shortlist = tools.shortlist_jobs("analytics", profile, {"seen"})
    assert shortlist[0] == wanted
    assert seen not in shortlist and senior not in shortlist and specialist not in shortlist
    assert len(shortlist) == config.REVIEW_LIMIT


def test_role_shortlist_recovers_specific_title_outside_dense_candidates(monkeypatch):
    specialist = Job("fit", "Franchise Support Consultant", "", "CONSULTANT")
    unqualified = Job("senior", "Senior Franchise Consultant", "", "CONSULTANT")
    dense = Job("dense", "Account Manager", "", "SALES")
    previous = Job("prior", "Franchise Specialist", "", "CONSULTANT")
    monkeypatch.setattr(tools, "_jobs", lambda: {j.job_id: j for j in (specialist, unqualified, dense, previous)})

    def dense_search(query, profile, exclude):
        assert exclude == {"prior", "fit"}
        return [dense]

    monkeypatch.setattr(tools, "shortlist_jobs", dense_search)
    profile = Profile("APPAREL", roles=["Franchise Owner"], early_career=True)
    assert tools.role_shortlist("Franchise Owner", profile, {"prior"}) == [specialist, dense]
