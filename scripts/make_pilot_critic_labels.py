"""Build weak Critic labels for model training with the existing dataset.

The labels describe observable skill and category evidence. Use independent
human judgments to evaluate job relevance.
"""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import random

from hiremind import config, matching, tools
from hiremind.roles import critic, searcher
from hiremind.schemas import Job, Profile
from hiremind.skills import extract_skills


@lru_cache(maxsize=20000)
def _listed_requirements(description: str) -> tuple[str, tuple[str, ...]]:
    text = matching._requirements(description)
    return text, tuple(extract_skills(text))


def critic_score(profile: Profile, job: Job) -> tuple[float, str] | None:
    """Score duties and qualifications using the evidence that the rules can see."""
    text, required = _listed_requirements(job.description)
    available = {skill.casefold() for skill in profile.skills}
    compared = required or tuple(job.skills)
    if not compared:
        return None
    shared = [skill for skill in compared if skill.casefold() in available]
    same_field = job.category == profile.category
    if required:
        # Use the scorer's evidence check to respect explicit alternatives.
        _, missing, _ = matching._requirement_evidence(text, {skill: skill for skill in available})
        coverage = (len(required) - len(missing)) / len(required)
    else:
        missing = []
        coverage = len(shared) / min(5, len(compared))
    score = 2 + 2 * same_field + 5 * min(1, coverage)
    missing_credentials = matching.missing_title_credentials(profile, job)
    if missing_credentials:
        score = min(score, 5.9)
    reason = f"The dataset field {'matches' if same_field else 'differs from'} the résumé field. "
    reason += ("Shared skills: " + ", ".join(sorted(set(shared))[:8]) + ". "
               if shared else "No shared listed skills were found. ")
    if missing:
        reason += "Check requirement skills without résumé evidence: " + ", ".join(missing[:6]) + ". "
    if missing_credentials:
        reason += "The résumé does not establish these title credentials: " + ", ".join(missing_credentials) + ". "
    return round(score, 1), reason.strip()


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=config.ROOT / "outputs" / "critic_labels")
    cli.add_argument("--train-limit", type=int)
    cli.add_argument("--val-limit", type=int)
    args = cli.parse_args()
    if any(limit is not None and limit < 1 for limit in (args.train_limit, args.val_limit)):
        cli.error("Split limits must be positive")
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        resumes = {row["id"]: row for row in csv.DictReader(handle)}
    train_ids, val_ids = (
        set((config.SPLITS_DIR / f"{split}.txt").read_text().splitlines())
        for split in ("train", "val")
    )
    if train_ids & val_ids:
        raise ValueError("Training and validation resume IDs overlap")

    import torch
    from hiremind import index

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    jobs = tools._jobs()
    job_index = index.load()
    by_category = {category: [job for job in jobs.values() if job.category == category] for category in config.CATEGORIES}
    other_category = {category: [job for job in jobs.values() if job.category != category] for category in config.CATEGORIES}
    rng = random.Random(config.SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    training_pairs = set()
    for split, ids, limit in (("train", train_ids, args.train_limit), ("val", val_ids, args.val_limit)):
        selected = sorted(ids & resumes.keys())
        if limit is not None and limit < len(selected):
            selected = rng.sample(selected, limit)
        profiles = []
        for resume_id in selected:
            row = resumes[resume_id]
            if row["category"] in by_category:
                profiles.append((resume_id, Profile(row["category"], skills=extract_skills(row["text"]))))
        vectors = index.embed([searcher.profile_queries(profile)[0] for _, profile in profiles])
        count = 0
        split_pairs = set()
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for (resume_id, profile), vector in zip(profiles, vectors, strict=True):
                dense = [jobs[job_id] for job_id, _ in job_index.search_vector(vector, 12)]
                same = rng.sample(by_category[profile.category], min(2, len(by_category[profile.category])))
                different = rng.sample(other_category[profile.category], min(2, len(other_category[profile.category])))
                neighbours = {job.job_id: job for job in [*same, *different, *dense]}
                written = 0
                for job in neighbours.values():
                    pair = (profile.category, tuple(profile.skills), job.job_id)
                    if pair in training_pairs or pair in split_pairs:
                        continue
                    judgment = critic_score(profile, job)
                    if judgment is None:
                        continue
                    score, reason = judgment
                    row = {
                        "task": "critic", "resume_id": resume_id, "job_id": job.job_id,
                        "prompt": critic.build_prompt(profile, job),
                        "completion": json.dumps({"score": score, "reason": reason}),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    split_pairs.add(pair)
                    written += 1
                    count += 1
                    if written == 12:
                        break
        if split == "train":
            training_pairs = split_pairs
        print(f"{split}: {count} weak Critic labels from {len(profiles)} résumés", flush=True)


if __name__ == "__main__":
    main()
