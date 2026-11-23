import json

from hiremind import config
from hiremind.memory import Memory
from hiremind.roles import searcher
from hiremind.schemas import Job, Profile
from scripts.make_searcher_traces import FEWER_THAN_FIVE, rule_agent

PROFILE = Profile(category="SALES", skills=["Microsoft Excel", "Customer service", "Negotiation", "CRM software"],
                  roles=["Sales Associate", "Store Manager"])


def job(i, category="SALES", skills=("Microsoft Excel", "Customer service")):
    return Job(job_id=str(i), title=f"Job {i}", description="text", category=category, skills=list(skills))


def tools_of(examples):
    return [json.loads(e["completion"])["tool"] for e in examples]


def fixed(results):
    """A search stand-in with the tools.shortlist_jobs signature."""
    return lambda query, profile, exclude: [j for j in results if j.job_id not in (exclude or set())]


def test_queries_follow_the_agent_fallback_order():
    examples = rule_agent(PROFILE, fixed([job(i) for i in range(6)]), score=lambda p, j: (8.0, "good"))
    first = json.loads(examples[0]["completion"])["arguments"]["query"]
    assert first == searcher.profile_queries(PROFILE)[0]


def test_good_first_search_is_search_then_stop():
    examples = rule_agent(PROFILE, fixed([job(i) for i in range(6)]), score=lambda p, j: (8.0, "good"))
    assert tools_of(examples) == ["search_jobs", "stop"]


def test_off_field_results_are_filtered_to_the_resume_category():
    results = [job(1), job(2, "CHEF"), job(3, "CHEF"), job(4, "HR")]
    examples = rule_agent(PROFILE, fixed(results), score=lambda p, j: (8.0, "good"))
    assert tools_of(examples)[:3] == ["search_jobs", "filter_jobs", "stop"]
    assert json.loads(examples[1]["completion"])["arguments"] == {"category": "SALES", "level": None}


def test_weak_results_lead_to_feedback_rejections_and_a_new_query():
    calls = []
    def search(query, profile, exclude):
        calls.append(query)
        return [job(len(calls) * 10 + i) for i in range(3)]
    examples = rule_agent(PROFILE, search, score=lambda p, j: (2.0, "weak"))
    assert tools_of(examples) == ["search_jobs", "stop"] * config.MAX_SEARCH_ROUNDS
    assert len(set(calls)) == len(calls) == config.MAX_SEARCH_ROUNDS  # never repeats a query
    second_search_prompt = examples[2]["prompt"]
    assert "No strong match found" in second_search_prompt and calls[0] in second_search_prompt
    assert "Requirements score 2/10 fell below the cutoff." in second_search_prompt  # the agent's rejection wording


def test_fewer_than_five_good_jobs_searches_again():
    calls = []
    def search(query, profile, exclude):
        calls.append(query)
        return [job(len(calls) * 10 + i) for i in range(2)]
    examples = rule_agent(PROFILE, search, score=lambda p, j: (8.0, "good"))
    assert len(calls) > 1 and FEWER_THAN_FIVE in examples[2]["prompt"]


def test_seen_jobs_are_excluded_from_later_searches():
    excluded_sets = []
    def search(query, profile, exclude):
        excluded_sets.append(set(exclude))
        return [job(1), job(2)]
    rule_agent(PROFILE, search, score=lambda p, j: (2.0, "weak"))
    assert excluded_sets[0] == set() and excluded_sets[1] == {"1", "2"}


def test_first_prompt_matches_the_runtime_searcher():
    examples = rule_agent(PROFILE, fixed([job(1)]), score=lambda p, j: (8.0, "good"))
    assert examples[0]["prompt"] == searcher.build_prompt(Memory(semantic=PROFILE))


def test_actions_pass_the_runtime_validation(monkeypatch):
    examples = rule_agent(PROFILE, fixed([job(1), job(2, "CHEF"), job(3, "HR")]), score=lambda p, j: (2.0, "weak"))
    from hiremind import slm
    for example in examples:
        monkeypatch.setattr(slm, "generate", lambda task, prompt, max_new_tokens=0, answer=example["completion"]: answer)
        name, args = searcher.next_action(Memory(semantic=PROFILE))
        assert name == json.loads(example["completion"])["tool"]
