"""Run a simple rule-based agent on the resumes and save every decision it makes.

Run as: python -m scripts.make_searcher_traces

The rule agent follows the same loop as hiremind.agent.run: each round it
searches, optionally filters to the resume's category, then stops; the Critic
rule (scripts.make_labels.critic_score) scores the candidates, and if none
reaches the cutoff it adds the Critic's feedback and searches again with a new
query, for at most config.MAX_SEARCH_ROUNDS rounds.

Each decision becomes one training example: the exact prompt the Searcher would
see at that point (searcher.build_prompt) and the action JSON it should return.
Only training and validation resumes are used. Outputs
data/processed/training/searcher_{train,val}.jsonl.
"""
import json

import pandas as pd

from hiremind import config
from hiremind.memory import Memory
from hiremind.roles import critic, searcher
from hiremind.schemas import Job, Judgement, Profile, Step
from scripts.make_labels import OUT_DIR, critic_score, parser_labels, profile_from_labels

MAX_QUERY_WORDS = 8


def _query(words: list[str]) -> str:
    """Join words, dropping repeats (case-insensitive), up to the Searcher's 8-word limit."""
    seen, kept = set(), []
    for word in " ".join(words).replace("/", " ").split():
        if word.casefold() not in seen:
            seen.add(word.casefold())
            kept.append(word)
    return " ".join(kept[:MAX_QUERY_WORDS])


def query_options(profile: Profile) -> list[str]:
    """Queries in the order the rule agent tries them: each role with the top
    skills, then the category with the top skills, then the category with the next skills."""
    field = profile.category.replace("-", " ").title()
    options = [_query([role, *profile.skills[:3]]) for role in profile.roles]
    options += [_query([field, *profile.skills[:3]]), _query([field, *profile.skills[3:6]]), _query([field])]
    return list(dict.fromkeys(q for q in options if q))


def _action(tool: str, arguments: dict) -> str:
    return json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))


def rule_agent(profile: Profile, search, score=critic_score) -> list[dict]:
    """Return the (prompt, completion) examples for one profile.

    `search(query)` returns a list of Job; `score(profile, job)` returns (score, reason).
    """
    memory = Memory(semantic=profile)
    examples = []

    def decide(tool: str, arguments: dict) -> None:
        examples.append({"prompt": searcher.build_prompt(memory), "completion": _action(tool, arguments)})

    for round_number in range(config.MAX_SEARCH_ROUNDS):
        tried = {q.casefold() for q in memory.episodic.queries}
        query = next((q for q in query_options(profile) if q.casefold() not in tried), None)
        if query is None:
            break
        decide("search_jobs", {"query": query})
        found = [job for job in search(query) if job.job_id not in memory.episodic.seen_job_ids]
        memory.episodic.queries.append(query)
        memory.episodic.seen_job_ids.update(job.job_id for job in found)
        memory.short_term.append(Step(state={}, action="search_jobs", args={"query": query},
                                      result=[{"job_id": job.job_id, "title": job.title} for job in found]))
        candidates = found

        # Narrow to the resume's field when most results are from other fields.
        in_field = [job for job in candidates if job.category == profile.category]
        if in_field and len(in_field) < len(candidates) / 2:
            arguments = {"category": profile.category, "level": None}
            decide("filter_jobs", arguments)
            candidates = in_field
            memory.short_term.append(Step(state={}, action="filter_jobs", args=arguments,
                                          result=[{"job_id": job.job_id, "title": job.title} for job in candidates]))

        decide("stop", {})
        memory.short_term.append(Step(state={}, action="stop", args={}))

        judgements = [Judgement(job.job_id, *score(profile, job)) for job in candidates]
        advice = critic.feedback(profile, judgements)
        if advice is None or round_number + 1 == config.MAX_SEARCH_ROUNDS:
            break
        memory.episodic.critic_feedback.append(advice)
        memory.episodic.rejected.update({j.job_id: j.reason for j in judgements if j.score < config.CRITIC_CUTOFF})
    return examples


def main() -> None:
    from hiremind import tools

    resumes = pd.read_csv(config.PROCESSED_DIR / "resumes.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        ids = set(map(int, (config.SPLITS_DIR / f"{split}.txt").read_text().split()))
        rows, rounds = [], []
        for resume in resumes[resumes["id"].isin(ids)].sort_values("id").to_dict("records"):
            profile = profile_from_labels(parser_labels(resume), resume["text"])
            examples = rule_agent(profile, tools.search_jobs)
            rounds.append(sum(json.loads(e["completion"])["tool"] == "search_jobs" for e in examples))
            rows += [{"task": "searcher", **e, "resume_id": resume["id"], "step": k} for k, e in enumerate(examples)]
        with (OUT_DIR / f"searcher_{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"searcher_{split}.jsonl: {len(rows)} examples from {len(rounds)} resumes; "
              f"search rounds per resume: {pd.Series(rounds).value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
