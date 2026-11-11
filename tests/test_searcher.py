import json

import pytest

from hiremind import slm, tools
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
    monkeypatch.setattr(tools, "search_jobs", lambda query: [job])
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
    monkeypatch.setattr(tools, "search_jobs", lambda **kwargs: pytest.fail("invalid action executed"))
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
    monkeypatch.setattr(tools, "search_jobs", lambda query: [job])
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
