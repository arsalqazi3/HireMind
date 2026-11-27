"""Check Parser categories on a validation sample or the hand-checked test cohort.

Run as: python -m scripts.validate_parser
This compares only category with the noisy dataset label, not full profile accuracy.
"""
import argparse
import csv
import json
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


def checked_test_resumes(path: Path) -> list[dict]:
    test_ids = set((config.SPLITS_DIR / "test.txt").read_text().splitlines())
    with path.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    ids = {row["resume_id"] for row in labels}
    if len(ids) != len(labels) or ids - test_ids:
        raise ValueError("Checked labels must contain unique test résumé IDs")
    with (config.PROCESSED_DIR / "resumes.csv").open(newline="", encoding="utf-8") as handle:
        resumes = {row["id"]: row for row in csv.DictReader(handle) if row["id"] in ids}
    selected = []
    for label in labels:
        row = resumes.get(label["resume_id"])
        checked, in_pool = label["checked_category"], label["in_job_pool"]
        if row is None or row["category"] != label["dataset_category"]:
            raise ValueError(f"Checked label does not match dataset résumé {label['resume_id']}")
        if in_pool == "1" and checked in config.CATEGORIES:
            selected.append({**row, "checked_category": checked})
        elif in_pool != "0" or checked != "NONE":
            raise ValueError(f"Invalid checked category for {label['resume_id']}")
    if not selected:
        raise ValueError("No in-pool résumés have checked categories")
    return selected


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--offset", type=int, default=0)
    arguments.add_argument("--count", type=int, default=1, help="Number of resumes per category")
    arguments.add_argument("--adapter", type=Path, help="Compare the base model and this Parser adapter")
    arguments.add_argument("--exclude-ids", type=Path, help="File of resume IDs used for training or checkpoint selection")
    arguments.add_argument("--checked-labels", type=Path, help="Hand-checked category CSV for the test split")
    arguments.add_argument("--output", type=Path, help="Save category predictions and counts as JSON")
    args = arguments.parse_args()
    if args.count < 1:
        arguments.error("--count must be positive")
    excluded = set(args.exclude_ids.read_text().splitlines()) if args.exclude_ids else set()
    if args.checked_labels:
        if args.offset or args.count != 1 or excluded:
            arguments.error("Checked-label evaluation uses the full in-pool cohort")
        rows = checked_test_resumes(args.checked_labels)
    else:
        rows = [row for i in range(args.count) for row in validation_resumes(args.offset + i, excluded)]
    modes = {"base": "", "adapter": str(args.adapter)} if args.adapter else {"base": ""}
    results = {}
    for mode, adapter in modes.items():
        predictions = {}
        with patch.dict(os.environ, {"HIREMIND_ADAPTER_DIR": adapter, "HIREMIND_ADAPTER_ROLES": "parser"}):
            for row in rows:
                started = time.perf_counter()
                try:
                    profile = parser.parse(row["text"])
                    prediction = profile.category
                except ValueError:
                    prediction = "invalid"
                predictions[row["id"]] = prediction
                print(mode, row["category"], "->", prediction, "seconds", round(time.perf_counter() - started, 1), flush=True)
        valid = sum(category != "invalid" for category in predictions.values())
        correct = sum(predictions[row["id"]] == row["category"] for row in rows)
        print(f"{mode}: {correct}/{len(rows)} category matches, {valid}/{len(rows)} valid profiles", flush=True)
        summary = {"total": len(rows), "valid_profiles": valid, "dataset_matches": correct}
        if args.checked_labels:
            checked = sum(predictions[row["id"]] == row["checked_category"] for row in rows)
            summary["checked_matches"] = checked
            print(f"{mode}: {checked}/{len(rows)} hand-checked category matches", flush=True)
        results[mode] = {"summary": summary, "predictions": predictions}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
