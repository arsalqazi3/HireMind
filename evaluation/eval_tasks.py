"""Per-role evaluation on the test set.

- Parser: category accuracy, macro F1, skill precision/recall/F1, valid JSON rate.
- Critic: MAE against the rule scores, rank agreement, reason checks.
- Searcher: valid and correct tool calls, stops at the right time, repeats, rounds used.

Each one is compared for the fine-tuned SLM, the base SLM, and (for the
category only) TF-IDF plus Logistic Regression from evaluation/baselines.py.
"""


def eval_parser(model) -> dict:
    raise NotImplementedError


def eval_critic(model) -> dict:
    raise NotImplementedError


def eval_searcher(model) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
