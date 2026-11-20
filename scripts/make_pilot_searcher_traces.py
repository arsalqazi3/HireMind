"""Build Searcher demonstrations from rule-based queries over the existing index."""
import argparse
import csv
import json
from pathlib import Path
import random

from hiremind import config, tools
from hiremind.memory import Memory
from hiremind.roles import searcher
from hiremind.schemas import Profile, Step
from hiremind.skills import extract_skills


def rule_agent(profile: Profile):
    """Demonstrate an initial search and a valid next action after tool results."""
    memory = Memory(semantic=profile)
    for query in searcher.profile_queries(profile)[:3]:
        args = {"query": query}
        yield searcher.build_prompt(memory), {"tool": "search_jobs", "arguments": args}
        jobs = tools.shortlist_jobs(query, profile, memory.episodic.seen_job_ids)
        memory.episodic.queries.append(query)
        memory.episodic.seen_job_ids.update(job.job_id for job in jobs)
        memory.short_term.append(Step(
            state={}, action="search_jobs", args=args,
            result=[{"job_id": job.job_id, "title": job.title} for job in jobs],
        ))
        if len(jobs) >= config.TOP_K:
            break
    yield searcher.build_prompt(memory), {"tool": "stop", "arguments": {}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=config.ROOT / "outputs" / "searcher_traces")
    cli.add_argument("--train-limit", type=int, default=100)
    cli.add_argument("--val-limit", type=int, default=30)
    args = cli.parse_args()
    if args.train_limit < 1 or args.val_limit < 1:
        cli.error("Split limits must be positive")
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        resumes = {row["id"]: row for row in csv.DictReader(handle)}
    train_ids, val_ids = (
        set((config.SPLITS_DIR / f"{split}.txt").read_text().splitlines())
        for split in ("train", "val")
    )
    if train_ids & val_ids:
        raise ValueError("Training and validation resume IDs overlap")

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.SEED)
    training_prompts = set()
    for split, ids, limit in (("train", train_ids, args.train_limit), ("val", val_ids, args.val_limit)):
        candidates = sorted(ids & resumes.keys())
        selected = rng.sample(candidates, min(limit, len(candidates)))
        count = 0
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            split_prompts = set()
            for resume_id in selected:
                row = resumes[resume_id]
                profile = Profile(category=row["category"], skills=extract_skills(row["text"]))
                for prompt, action in rule_agent(profile):
                    if prompt in training_prompts or prompt in split_prompts:
                        continue
                    handle.write(json.dumps({
                        "task": "searcher",
                        "prompt": prompt,
                        "completion": json.dumps(action),
                        "resume_id": resume_id,
                    }) + "\n")
                    split_prompts.add(prompt)
                    count += 1
        if split == "train":
            training_prompts = split_prompts
        print(f"{split}: {count} Searcher examples from {len(selected)} résumés", flush=True)


if __name__ == "__main__":
    main()
