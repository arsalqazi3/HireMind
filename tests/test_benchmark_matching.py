from hiremind import agent
from hiremind.schemas import Judgement, Profile
from scripts import benchmark_matching


def test_benchmark_counts_repeated_scoring_rounds_without_saving_source_text(monkeypatch):
    events = [
        {"phase": "parsing"}, {"phase": "searching", "round": 1},
        {"phase": "scoring", "round": 1, "completed": 0},
        {"phase": "scoring", "round": 1, "completed": 3},
        {"phase": "searching", "round": 2}, {"phase": "scoring", "round": 2, "completed": 2},
        {"phase": "done"},
    ]

    def run(text, *, progress, in_depth):
        assert text == "source résumé" and in_depth
        for event in events:
            progress(event)
        return agent.Result(Profile("IT"), [Judgement("j", 8, "reason")], retries=1)

    monkeypatch.setattr(agent, "run", run)
    monkeypatch.setattr(agent, "cover_letter", lambda profile, job_id: "source letter")
    monkeypatch.setattr(benchmark_matching, "peak_rss_mib", lambda: 300.0)
    result = benchmark_matching.benchmark_case({"id": "r", "text": "source résumé"}, in_depth=True)
    assert result["job_ids"] == ["j"]
    assert result["reviewed_jobs_by_round"] == {1: 3, 2: 2}
    assert result["phase_seconds"].keys() == {"parsing", "searching", "scoring"}
    assert result["retries"] == 1
    assert result["cover_letter_seconds"] >= 0
    assert "source" not in str(result)
