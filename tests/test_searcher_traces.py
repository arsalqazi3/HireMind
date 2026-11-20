import json

from hiremind import config
from hiremind.memory import Memory
from hiremind.roles import searcher
from hiremind.schemas import Job, Profile
from scripts.make_searcher_traces import query_options, rule_agent

PROFILE = Profile(category="SALES", skills=["Microsoft Excel", "Customer service", "Negotiation", "CRM software"],
                  roles=["Sales Associate", "Store Manager"])


def job(i, category="SALES", skills=("Microsoft Excel", "Customer service")):
    return Job(job_id=str(i), title=f"Job {i}", description="text", category=category, skills=list(skills))


def tools_of(examples):
    return [json.loads(e["completion"])["tool"] for e in examples]


def test_queries_are_short_unique_and_role_first():
    options = query_options(PROFILE)
    assert options[0] == "Sales Associate Microsoft Excel Customer service Negotiation"
    long_role = Profile(category="SALES", skills=PROFILE.skills, roles=["Senior Regional Retail Sales Store Manager"])
    assert query_options(long_role)[0] == "Senior Regional Retail Sales Store Manager Microsoft Excel"  # cut at 8 words
    assert all(len(q.split()) <= 8 for q in options)
    assert len(options) == len(set(options))
    assert any(q.startswith("Sales") and "Negotiation" not in q for q in options)


def test_good_first_search_is_search_then_stop():
    examples = rule_agent(PROFILE, lambda q: [job(i) for i in range(5)], score=lambda p, j: (8.0, "good"))
    assert tools_of(examples) == ["search_jobs", "stop"]


def test_off_field_results_are_filtered_to_the_resume_category():
    results = [job(1), job(2, "CHEF"), job(3, "CHEF"), job(4, "HR")]
    examples = rule_agent(PROFILE, lambda q: results, score=lambda p, j: (8.0, "good"))
    assert tools_of(examples) == ["search_jobs", "filter_jobs", "stop"]
    assert json.loads(examples[1]["completion"])["arguments"] == {"category": "SALES", "level": None}


def test_weak_results_lead_to_feedback_and_a_new_query():
    calls = []
    def search(query):
        calls.append(query)
        return [job(len(calls) * 10 + i) for i in range(3)]
    examples = rule_agent(PROFILE, search, score=lambda p, j: (2.0, "weak"))
    assert tools_of(examples) == ["search_jobs", "stop"] * config.MAX_SEARCH_ROUNDS
    assert len(set(calls)) == len(calls) == config.MAX_SEARCH_ROUNDS  # never repeats a query
    second_search_prompt = examples[2]["prompt"]
    assert "No strong match found" in second_search_prompt and calls[0] in second_search_prompt


def test_first_prompt_matches_the_runtime_searcher():
    examples = rule_agent(PROFILE, lambda q: [job(1)], score=lambda p, j: (8.0, "good"))
    assert examples[0]["prompt"] == searcher.build_prompt(Memory(semantic=PROFILE))


def test_actions_pass_the_runtime_validation(monkeypatch):
    examples = rule_agent(PROFILE, lambda q: [job(1), job(2, "CHEF"), job(3, "HR")], score=lambda p, j: (2.0, "weak"))
    from hiremind import slm
    for example in examples:
        monkeypatch.setattr(slm, "generate", lambda task, prompt, max_new_tokens=0, answer=example["completion"]: answer)
        name, args = searcher.next_action(Memory(semantic=PROFILE))
        assert name == json.loads(example["completion"])["tool"]
