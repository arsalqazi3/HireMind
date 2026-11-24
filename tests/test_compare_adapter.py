from evaluation.compare_adapter import parse_json, score, set_f1


def test_json_is_found_inside_a_code_fence():
    assert parse_json('```json\n{"score": 7}\n```') == {"score": 7}
    assert parse_json("no json here") is None
    assert parse_json('{"cut off": ') is None


def test_parser_scoring():
    target = '{"category": "SALES", "years_experience": null, "education": ["MSc"], "roles": ["Recruiter"], "projects": []}'
    exact = score("parser", target, target)
    assert exact == {"valid": True, "category": True, "years": True, "roles_f1": 1.0, "education_f1": 1.0}
    assert score("parser", "{broken", target)["valid"] is False


def test_critic_and_searcher_scoring():
    assert score("critic", '{"score": 6, "reason": "x"}', '{"score": 4.0, "reason": "y"}') == \
        {"valid": True, "abs_error": 2.0, "within_1": False}
    assert score("critic", "seven", '{"score": 4.0}')["valid"] is False
    target = '{"tool":"stop","arguments":{}}'
    assert score("searcher", '{"tool": "stop", "arguments": {}}', target) == {"valid": True, "tool": True, "exact": True}
    assert score("searcher", '{"tool": "delete"}', target)["valid"] is False


def test_set_f1_ignores_case():
    assert set_f1(["Recruiter", "Analyst"], ["recruiter"]) == 2 / 3
    assert set_f1([], []) == 1.0
