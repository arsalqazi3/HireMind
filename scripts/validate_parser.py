"""Check base-model parsing on five validation resumes from different fields.

Run as: python -m scripts.validate_parser
This compares only category with the noisy dataset label, not full profile accuracy.
"""
import argparse
import csv
import os
from pathlib import Path
import time
from unittest.mock import patch

from hiremind import config
from hiremind.roles import parser

CATEGORIES = ("ACCOUNTANT", "CHEF", "HEALTHCARE", "INFORMATION-TECHNOLOGY", "TEACHER")


def validation_resumes(offset: int = 0, exclude: set[str] | None = None):
    if offset < 0:
        raise ValueError("offset must not be negative")
    ids = set((config.SPLITS_DIR / "val.txt").read_text().splitlines())
    ids -= exclude or set()
    selected = {category: [] for category in CATEGORIES}
    with (config.PROCESSED_DIR / "resumes.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] in ids and row["category"] in CATEGORIES:
                selected[row["category"]].append(row)
    missing = {category for category, rows in selected.items() if len(rows) <= offset}
    if missing:
        raise ValueError(f"Missing validation categories: {sorted(missing)}")
    return [selected[category][offset] for category in CATEGORIES]


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--offset", type=int, default=0)
    arguments.add_argument("--count", type=int, default=1, help="Number of resumes per category")
    arguments.add_argument("--adapter", type=Path, help="Compare the base model and this Parser adapter")
    arguments.add_argument("--exclude-ids", type=Path, help="File of resume IDs used for training or checkpoint selection")
    args = arguments.parse_args()
    if args.count < 1:
        arguments.error("--count must be positive")
    excluded = set(args.exclude_ids.read_text().splitlines()) if args.exclude_ids else set()
    rows = [row for i in range(args.count) for row in validation_resumes(args.offset + i, excluded)]
    modes = {"base": "", "adapter": str(args.adapter)} if args.adapter else {"base": ""}
    for mode, adapter in modes.items():
        correct = valid = 0
        with patch.dict(os.environ, {"HIREMIND_ADAPTER_DIR": adapter, "HIREMIND_ADAPTER_ROLES": "parser"}):
            for row in rows:
                started = time.perf_counter()
                try:
                    profile = parser.parse(row["text"])
                    valid += 1
                    correct += profile.category == row["category"]
                    prediction = profile.category
                except ValueError:
                    prediction = "invalid"
                print(mode, row["category"], "->", prediction, "seconds", round(time.perf_counter() - started, 1), flush=True)
        print(f"{mode}: {correct}/{len(rows)} category matches, {valid}/{len(rows)} valid profiles", flush=True)


if __name__ == "__main__":
    main()
