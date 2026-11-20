"""Build a compact four-role set from the current splits for a mixed LoRA pilot."""
import argparse
import csv
import json
from pathlib import Path
import random
import re

from hiremind import config, tools
from hiremind.roles import parser, writer
from scripts.clean_data import map_job_category


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def profile_target(row: dict):
    profile = parser._build_profile(row["text"], row["category"], {})
    if not profile.roles:
        headline = re.split(r"\b(?:summary|highlights|professional profile|education|skills)\b", row["text"], maxsplit=1, flags=re.I)[0]
        headline = headline.strip()
        if len(headline.split()) <= 9 and map_job_category(headline) == row["category"]:
            profile.roles = [headline]
    # Keep the noisy dataset category as supervision and leave missing facts empty.
    profile.category = row["category"]
    profile.name = None
    target = {k: getattr(profile, k) for k in ("category", "skills", "years_experience", "education", "roles", "projects")}
    return profile, target


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--searcher", type=Path, default=config.ROOT / "outputs/searcher_traces")
    cli.add_argument("--critic", type=Path, default=config.ROOT / "outputs/critic_labels")
    cli.add_argument("--output", type=Path, default=config.ROOT / "outputs/joint_training")
    cli.add_argument("--train-per-role", type=int, default=48)
    cli.add_argument("--val-per-role", type=int, default=12)
    args = cli.parse_args()
    if min(args.train_per_role, args.val_per_role) < 1:
        cli.error("Split limits must be positive")
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        resumes = {row["id"]: row for row in csv.DictReader(handle)}
    train_ids, val_ids = (
        set((config.SPLITS_DIR / f"{name}.txt").read_text().splitlines()) for name in ("train", "val")
    )
    if train_ids & val_ids:
        raise ValueError("Training and validation IDs overlap")
    rng = random.Random(config.SEED)
    training_prompts = set()
    args.output.mkdir(parents=True, exist_ok=True)
    for split, ids, limit in (("train", train_ids, args.train_per_role), ("val", val_ids, args.val_per_role)):
        critic_rows, searcher_rows = (rows(folder / f"{split}.jsonl") for folder in (args.critic, args.searcher))
        if any(row["resume_id"] not in ids for row in [*critic_rows, *searcher_rows]):
            raise ValueError(f"Source JSONL contains résumé IDs outside the {split} split")
        candidates = sorted(i for i in ids & resumes.keys() if len(resumes[i]["text"].split()) < 700)
        rng.shuffle(candidates)
        parser_rows, writer_rows = [], []
        for resume_id in candidates[:limit]:
            row = resumes[resume_id]
            _, target = profile_target(row)
            parser_rows.append({"task": "parser", "resume_id": resume_id,
                                "prompt": parser.build_prompt(row["text"]), "completion": json.dumps(target)})
        rng.shuffle(critic_rows)
        writer_ids = set()
        for row in critic_rows:
            resume_id = row["resume_id"]
            if resume_id in writer_ids or json.loads(row["completion"])["score"] < 6:
                continue
            profile, _ = profile_target(resumes[resume_id])
            job = tools.get_job_details(row["job_id"])
            if profile.category != job.category or not tools.skill_overlap(profile, job):
                continue
            writer_rows.append({"task": "writer", "resume_id": resume_id, "prompt": writer.build_prompt(profile, job),
                                "completion": writer.write_checked(profile, job)})
            writer_ids.add(resume_id)
            if len(writer_rows) >= limit:
                break
        rng.shuffle(searcher_rows)
        seen = set(training_prompts)
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for group in (parser_rows, searcher_rows, critic_rows, writer_rows):
                written = 0
                for row in group:
                    key = row["task"], row["prompt"]
                    if key in seen:
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    seen.add(key)
                    written += 1
                    if written == limit:
                        break
                if not written:
                    raise ValueError(f"Cannot build a non-empty role set for {split}")
                print(f"{split} {group[0]['task']}: {written} examples", flush=True)
        if split == "train":
            training_prompts = seen


if __name__ == "__main__":
    main()
