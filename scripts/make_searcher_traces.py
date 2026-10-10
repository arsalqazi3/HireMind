"""Run a simple rule-based agent on the training resumes and save every step.

Run as: python -m scripts.make_searcher_traces

Each step (state, action, tool result, feedback) is saved in the
hiremind.schemas.Step format so the SLM can learn the Searcher role from them.
"""


def rule_agent(profile: dict) -> list[dict]:
    """Return the list of steps the rule-based agent takes for one profile."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
