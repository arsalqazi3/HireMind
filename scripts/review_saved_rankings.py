"""Resume automated review of saved blind ranking packets with the local reviewer."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import random

from tokenizers import Tokenizer

from evaluation import ablation, ai_reviewer
from hiremind import config, tools


ROOT = Path("outputs/evaluation_2026-11-24")
DEPTH_ASSESSMENTS = Path("outputs/depth_study/assessments.json")
SOURCES = {
    "validation": ROOT / "validation_blind_rows.json",
    "historical": ROOT / "historical_blind_rows.json",
}
SNAPSHOTS = {
    "validation": ROOT / "validation_step800.json",
    "historical": Path("outputs/ablation_test_joint.json"),
}


def blind_rows(cohort: str) -> list[dict]:
    if SOURCES[cohort].is_file():
        return json.loads(SOURCES[cohort].read_text(encoding="utf-8"))

    snapshot = json.loads(SNAPSHOTS[cohort].read_text(encoding="utf-8"))
    pool = ablation.review_pool(snapshot)
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        resumes = {row["id"]: row["text"] for row in csv.DictReader(handle) if row["id"] in pool}
    rng = random.Random(config.SEED)
    rows = []
    for resume_id, job_ids in pool.items():
        ordered = sorted(job_ids)
        rng.shuffle(ordered)
        for job_id in ordered:
            job = tools.get_job_details(job_id)
            rows.append({"resume_id": resume_id, "job_id": job_id, "resume": resumes[resume_id],
                         "title": job.title, "description": job.description})
    return rows


def write_labels(path: Path, rows: list[dict], reviews: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("resume_id", "job_id", "relevant", "source", "note"))
        for row in rows:
            review = reviews.get(row["resume_id"], {})
            judgment = next((job for job in review.get("jobs", [])
                             if job["job_id"] == row["job_id"]), None)
            if judgment is None:
                writer.writerow((row["resume_id"], row["job_id"], "", "", ""))
                continue
            grade = judgment["grade"]
            relevant = "" if grade < 0 else int(grade >= 2)
            writer.writerow((row["resume_id"], row["job_id"], relevant,
                             judgment.get("source", review["source"]), judgment["reason"]))
    temporary.replace(path)


def historical_seeds() -> dict[str, dict[str, dict]]:
    seeds = defaultdict(dict)
    depth = (json.loads(DEPTH_ASSESSMENTS.read_text(encoding="utf-8"))
             if DEPTH_ASSESSMENTS.is_file() else {})
    for resume_id, review in depth.items():
        for job in review["jobs"]:
            seeds[resume_id][job["job_id"]] = {**job, "source": review["source"]}
    starter = ROOT / "ai_review/test_starter_review.csv"
    if not starter.is_file():
        return seeds
    with starter.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            answer = row["relevant"]
            grade = -1 if answer == "" else (2 if answer == "1" else 1)
            seeds[row["resume_id"]][row["job_id"]] = {
                "job_id": row["job_id"], "grade": grade,
                "reason": row["note"], "source": row["source"],
            }
    return seeds


def score_checkpoint(cohort: str, output: Path, reviews: dict) -> None:
    snapshot = json.loads(SNAPSHOTS[cohort].read_text(encoding="utf-8"))
    pool = ablation.review_pool(snapshot)
    judgments = {}
    uncertain = 0
    for resume_id, review in reviews.items():
        jobs = {job["job_id"]: job["grade"] for job in review["jobs"]}
        if len(jobs) != len(review["jobs"]) or jobs.keys() != pool.get(resume_id):
            raise ValueError(f"Checkpoint does not match the ranking pool for résumé {resume_id}")
        judgments[resume_id] = {job_id: grade >= 2 for job_id, grade in jobs.items() if grade >= 0}
        uncertain += sum(grade < 0 for grade in jobs.values())
    groups = {"all": snapshot}
    if cohort == "historical":
        groups = ablation.split_checked_cohort(snapshot, Path("evaluation/labels/test_categories.csv"))
    baseline = "A" if cohort == "historical" else "D"
    result = {
        "reviewed_resumes": len(reviews),
        "total_resumes": len(pool),
        "judged_pairs": sum(map(len, judgments.values())),
        "uncertain_pairs": uncertain,
        "total_pairs": sum(map(len, pool.values())),
        "groups": {name: {"metrics": ablation.score(group, judgments),
                          "paired": ablation.compare_precision(group, judgments, baseline)}
                   for name, group in groups.items()},
    }
    ablation.save_json(output / "summary.json", result)
    print(f"Scored {len(reviews)}/{len(pool)} reviewed résumés in {output / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=SOURCES)
    parser.add_argument("--limit", type=int, help="Maximum new résumés to review")
    parser.add_argument("--score-only", action="store_true", help="Score one saved checkpoint without model inference")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    output = ROOT / "ai_review" / args.cohort
    reviews_path = output / "reviews.json"
    labels_path = output / "labels.csv"
    reviews = json.loads(reviews_path.read_text(encoding="utf-8")) if reviews_path.exists() else {}
    if args.score_only:
        if not reviews_path.is_file():
            parser.error("No saved review checkpoint is available")
        score_checkpoint(args.cohort, output, reviews)
        return

    rows = blind_rows(args.cohort)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["resume_id"]].append(row)
    seeds = historical_seeds() if args.cohort == "historical" else {}
    token_path = next((config.MODELS_DIR / "huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots").glob("*/tokenizer.json"))
    tokenizer = Tokenizer.from_file(str(token_path))

    completed = 0
    for resume_id, group in grouped.items():
        if resume_id in reviews:
            continue
        if args.limit is not None and completed >= args.limit:
            break
        expected = {row["job_id"] for row in group}
        prior = {job_id: job for job_id, job in seeds.get(resume_id, {}).items()
                 if job_id in expected}
        jobs = [{"job_id": row["job_id"], "title": row["title"],
                 "description": row["description"]} for row in group
                if row["job_id"] not in prior]
        fresh = []
        for start in range(0, len(jobs), 5):
            batch = jobs[start:start + 5]
            fresh.extend(ai_reviewer.judge(group[0]["resume"], batch, tokenizer)["jobs"])
        result = {"source": ai_reviewer.MODEL, "jobs": [*prior.values(), *fresh]}
        if {job["job_id"] for job in result["jobs"]} != expected or len(result["jobs"]) != len(expected):
            raise ValueError(f"Review coverage mismatch for résumé {resume_id}")
        reviews[resume_id] = result
        ablation.save_json(reviews_path, reviews)
        write_labels(labels_path, rows, reviews)
        completed += 1
        print(f"{len(reviews)}/{len(grouped)} {resume_id}: {len(jobs)} jobs reviewed", flush=True)
    write_labels(labels_path, rows, reviews)
    score_checkpoint(args.cohort, output, reviews)


if __name__ == "__main__":
    main()
