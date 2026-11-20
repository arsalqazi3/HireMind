"""Build train/validation data for a small category-only Parser LoRA pilot."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import random

from hiremind import config
from hiremind.roles.parser import build_prompt


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=config.ROOT / "outputs" / "parser_lora")
    args = cli.parse_args()
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        resumes = list(csv.DictReader(handle))
    train_ids, val_ids = (
        set((config.SPLITS_DIR / f"{split}.txt").read_text().splitlines())
        for split in ("train", "val")
    )
    if train_ids & val_ids:
        raise ValueError("Training and validation resume IDs overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    used_ids = []
    for split, ids, per_category in (("train", train_ids, 10), ("val", val_ids, 1)):
        groups = defaultdict(list)
        for row in resumes:
            if row["id"] in ids and len(row["text"].split()) < 850 and row["category"] in config.CATEGORIES:
                groups[row["category"]].append(row)
        rng = random.Random(config.SEED)
        count = 0
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for category, rows in sorted(groups.items()):
                for row in rng.sample(rows, min(per_category, len(rows))):
                    # Stop the loss after the category prefix. The remaining
                    # profile fields have no gold answers in this dataset.
                    example = {
                        "task": "parser",
                        "prompt": build_prompt(row["text"]),
                        "completion": '{"category": ' + json.dumps(category) + ',',
                        "partial": True,
                        "resume_id": row["id"],
                    }
                    handle.write(json.dumps(example) + "\n")
                    used_ids.append(row["id"])
                    count += 1
        print(f"{split}: {count} weakly labelled category examples")
    (args.output / "used_ids.txt").write_text("\n".join(used_ids) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
