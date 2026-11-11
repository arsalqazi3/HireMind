"""Check base-model parsing on five validation resumes from different fields.

Run as: python -m scripts.validate_parser
This compares only category with the noisy dataset label, not full profile accuracy.
"""
import argparse
import csv
import time

from hiremind import config
from hiremind.roles import parser

CATEGORIES = ("ACCOUNTANT", "CHEF", "HEALTHCARE", "INFORMATION-TECHNOLOGY", "TEACHER")


def validation_resumes(offset: int = 0):
    if offset < 0:
        raise ValueError("offset must not be negative")
    ids = set((config.SPLITS_DIR / "val.txt").read_text().splitlines())
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
    for row in validation_resumes(arguments.parse_args().offset):
        started = time.perf_counter()
        try:
            profile = parser.parse(row["text"])
            print(row["category"], "->", profile.category, "seconds", round(time.perf_counter() - started, 1), flush=True)
        except ValueError as exc:
            print(row["category"], "-> invalid:", str(exc), "seconds", round(time.perf_counter() - started, 1), flush=True)


if __name__ == "__main__":
    main()
