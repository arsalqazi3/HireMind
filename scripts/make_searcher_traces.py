"""Run a simple rule-based agent on the resumes and save every decision it makes.

Run as: python -m scripts.make_searcher_traces

The rule agent follows the loop in hiremind.agent.run and hiremind.roles.searcher:
each round it searches with tools.shortlist_jobs (excluding jobs already seen),
optionally filters to the resume's category, then stops. The Critic rule
(scripts.make_labels.critic_score) stands in for the Critic: jobs below the
cutoff are remembered as rejections with the agent's wording, and the agent's
feedback is added before the next round, for at most config.MAX_SEARCH_ROUNDS rounds.
Queries come from searcher.profile_queries, the agent's own fallback queries.

Each decision becomes one training example: the exact prompt the Searcher would
see at that point (searcher.build_prompt) and the action JSON it should return.
Only training and validation resumes are used. Outputs
data/processed/training/searcher_{train,val}.jsonl.
"""
import json

import pandas as pd

from hiremind import config, matching
from hiremind.memory import Memory
from hiremind.roles import critic, searcher
from hiremind.schemas import Judgement, Profile, Step
from scripts.make_labels import OUT_DIR, critic_score, parser_labels, profile_from_labels

FEWER_THAN_FIVE = "Fewer than five suitable jobs found; search for additional roles and skills."


def _action(tool: str, arguments: dict) -> str:
    return json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))


def rule_agent(profile: Profile, search, score=critic_score) -> list[dict]:
    """Return the (prompt, completion) examples for one profile.

    `search(query, profile, exclude)` returns a list of Job, like tools.shortlist_jobs;
    `score(profile, job)` returns (score, reason).
    """
    memory = Memory(semantic=profile)
    examples = []
    scored_ids: set[str] = set()
    eligible_ids: set[str] = set()

    def decide(tool: str, arguments: dict) -> None:
        examples.append({"prompt": searcher.build_prompt(memory), "completion": _action(tool, arguments)})

    for round_number in range(config.MAX_SEARCH_ROUNDS):
        tried = {q.casefold() for q in memory.episodic.queries}
        query = next((q for q in searcher.profile_queries(profile) if q.casefold() not in tried), None)
        if query is None:
            break
        decide("search_jobs", {"query": query})
        found = [job for job in search(query, profile, memory.episodic.seen_job_ids)
                 if job.job_id not in memory.episodic.seen_job_ids]
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

        jobs = [job for job in candidates if job.job_id not in scored_ids]
        if not jobs:
            break
        scored_ids.update(job.job_id for job in jobs)
        judgements = {job.job_id: Judgement(job.job_id, *score(profile, job)) for job in jobs}
        eligible = [j for j in judgements.values() if j.score >= config.CRITIC_CUTOFF]
        eligible = [j for j in eligible if not matching.early_seniority_conflict(profile, next(x for x in jobs if x.job_id == j.job_id))]
        eligible_ids.update(j.job_id for j in eligible)
        advice = critic.feedback(profile, eligible)
        if advice is None and len(eligible_ids) < config.TOP_K:
            advice = FEWER_THAN_FIVE
        if advice is None or round_number + 1 == config.MAX_SEARCH_ROUNDS:
            break
        memory.episodic.critic_feedback.append(advice)
        for job in jobs:
            if job.job_id in eligible_ids:
                continue
            if matching.early_seniority_conflict(profile, job):
                failure = "Seniority conflicts with the early-career search."
            else:
                failure = f"Requirements score {judgements[job.job_id].score:g}/10 fell below the cutoff."
            memory.episodic.rejected[job.job_id] = f"{job.title[:100]}: {failure}"
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
            examples = rule_agent(profile, tools.shortlist_jobs)
            rounds.append(sum(json.loads(e["completion"])["tool"] == "search_jobs" for e in examples))
            rows += [{"task": "searcher", **e, "resume_id": resume["id"], "step": k} for k, e in enumerate(examples)]
        with (OUT_DIR / f"searcher_{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"searcher_{split}.jsonl: {len(rows)} examples from {len(rounds)} resumes; "
              f"search rounds per resume: {pd.Series(rounds).value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
