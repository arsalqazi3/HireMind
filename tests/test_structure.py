from hiremind import config
from hiremind.memory import Memory
from hiremind.tools import TOOLS
from evaluation.ablation import VERSIONS


def test_four_roles_have_task_tokens():
    assert set(config.TASK_TOKENS) == {"parser", "searcher", "critic", "writer"}


def test_four_tools():
    assert set(TOOLS) == {"search_jobs", "filter_jobs", "get_job_details", "skill_overlap"}


def test_split_adds_up():
    assert abs(sum(config.SPLIT) - 1.0) < 1e-9


def test_memory_starts_empty():
    m = Memory()
    assert m.semantic is None
    assert m.episodic.queries == []
    assert m.short_term == []


def test_five_ablation_versions():
    assert list(VERSIONS) == ["A", "B", "C", "D", "E"]
