from evaluation.retrieval import reviewed_pairs, summarize


def test_sparse_judgments_bound_precision_and_preserve_unjudged_slots():
    result = summarize([(["yes", "unknown", "yes", "no"], {"yes": True, "no": False, "other": True})], 5)
    assert result["confirmed_relevant"] == 1
    assert result["confirmed_irrelevant"] == 1
    assert result["unjudged"] == 1
    assert result["judgment_coverage"] == 0.4
    assert result["precision_bounds"] == [0.2, 0.4]
    assert result["share_of_known_positives_retrieved"] == 0.5
    assert summarize([(["unknown"], {})], 5)["precision_bounds"] is None


def test_incomplete_drafted_or_conflicting_votes_do_not_count():
    assignments = [
        {"pair_id": code, "resume_id": "r", "job_id": code, "labellers": "a|b"}
        for code in ("complete", "draft", "conflict", "missing")
    ]
    votes = [
        {"pair_id": "complete", "labeller": "a", "relevant": "1"},
        {"pair_id": "complete", "labeller": "b", "relevant": "1"},
        {"pair_id": "draft", "labeller": "a", "relevant": "1"},
        {"pair_id": "draft", "labeller": "b", "relevant": "1", "note": "AI draft"},
        {"pair_id": "conflict", "labeller": "a", "relevant": "1"},
        {"pair_id": "conflict", "labeller": "b", "relevant": "0"},
        {"pair_id": "missing", "labeller": "a", "relevant": "1"},
    ]
    assert reviewed_pairs(assignments, votes) == {"r": {"complete": True}}
    assert reviewed_pairs(assignments, votes, allow_ai_drafts=True) == {"r": {"complete": True, "draft": True}}
