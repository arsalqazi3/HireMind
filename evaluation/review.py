"""Prepare and reconcile independent judgments for a frozen ranking snapshot."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from evaluation import ablation
from hiremind import config, tools


def freeze_batch(snapshot: dict, limit: int, versions: list[str] | None = None) -> dict:
    """Select the first fixed cohort entries after every requested version has run them."""
    ids = snapshot["settings"]["resume_ids"]
    if not 1 <= limit <= len(ids):
        raise ValueError(f"Choose a batch size from 1 to {len(ids)}")
    cohort = ids[:limit]
    runs = snapshot["runs"]
    requested = versions or list(runs)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Specify at least one version and list each version once")
    pending = {version: sum(resume_id not in runs.get(version, {}) for resume_id in cohort) for version in requested}
    if any(pending.values()):
        incomplete = ", ".join(f"{version} ({count} pending)" for version, count in pending.items() if count)
        raise ValueError(f"The fixed batch needs results from {incomplete}")
    return {"settings": {**snapshot["settings"], "resume_ids": cohort},
            "runs": {version: {resume_id: runs[version][resume_id] for resume_id in cohort} for version in requested}}


def prepare(snapshot, order_path: Path, output: Path):
    pool = ablation.review_pool(snapshot)
    keys = set()
    with order_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys.add((row["resume_id"], row["job_id"]))
    expected = {(resume, job) for resume, jobs in pool.items() for job in jobs}
    if keys != expected:
        raise ValueError("The blind packet does not match this ranking snapshot")
    with (config.PROCESSED_DIR / "resumes.csv").open(newline="", encoding="utf-8") as handle:
        resumes = {row["id"]: row["text"] for row in csv.DictReader(handle) if row["id"] in pool}
    output_rows = []
    with order_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            resume = row["resume_id"]
            job = tools.get_job_details(row["job_id"])
            output_rows.append({"resume_id": resume, "job_id": job.job_id, "title": job.title,
                                "resume": resumes[resume], "description": job.description})
    if len(output_rows) != len(expected):
        raise ValueError("The blind packet contains duplicate pairs")
    ablation.save_json(output, output_rows)


def read_workbook(path: Path, expected: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=False)
    try:
        values = book["Review"].values
        header = next((row for row in values if row[:2] == ("resume_id", "job_id")), ())
        if not header or "relevant" not in header or "note" not in header:
            raise ValueError(f"{path.name} has no review header")
        relevant, note = header.index("relevant"), header.index("note")
        votes = {}
        for row in values:
            if not any(value is not None for value in row):
                continue
            key = str(row[0]), str(row[1])
            raw_label = row[relevant]
            label = "" if raw_label is None else str(raw_label).strip()
            if key not in expected or key in votes or label not in {"", "0", "1"}:
                raise ValueError(f"{path.name}: unknown pair, duplicate or invalid judgment {key}")
            if label and "ai draft" in str(row[note] or "").casefold():
                raise ValueError(f"{path.name}: AI draft must be reviewed for {key}")
            votes[key] = label
        if votes.keys() != expected:
            raise ValueError(f"{path.name} is missing review pairs")
        return votes
    finally:
        book.close()


def combine(left: dict, right: dict, resolutions: dict | None = None) -> tuple[list, list]:
    if left.keys() != right.keys():
        raise ValueError("Reviewer pair sets differ")
    resolutions = resolutions or {}
    conflicts = {key for key, vote in left.items() if vote and right[key] and vote != right[key]}
    if resolutions.keys() - conflicts:
        raise ValueError("Adjudication can resolve only pairs with conflicting judgments")
    accepted, pending = [], []
    for resume, job in sorted(left):
        key = resume, job
        a, b = left[key], right[key]
        final = a if a and a == b else resolutions.get(key, "")
        accepted.append((resume, job, final))
        if not final:
            pending.append((resume, job, a, b, "conflict" if key in conflicts else "missing judgment", ""))
    return accepted, pending


def agreement(left: dict, right: dict) -> dict:
    """Measure reviewer agreement before adjudication, using completed pairs."""
    votes = Counter((vote, right[key]) for key, vote in left.items() if vote and right[key])
    total = sum(votes.values())
    agree = votes["0", "0"] + votes["1", "1"]
    result = {"pairs": len(left), "double_reviewed": total, "agreed": agree,
              "disagreed": total - agree, "agreement": None, "cohen_kappa": None}
    if total:
        observed = agree / total
        expected = sum(
            sum(n for (a, _), n in votes.items() if a == label)
            * sum(n for (_, b), n in votes.items() if b == label)
            for label in ("0", "1")
        ) / total**2
        result["agreement"] = observed
        if expected < 1:
            result["cohen_kappa"] = (observed - expected) / (1 - expected)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Save a fixed starter batch for blind review")
    freeze.add_argument("snapshot", type=Path)
    freeze.add_argument("output", type=Path)
    freeze.add_argument("--limit", type=int, required=True)
    freeze.add_argument("--versions", nargs="+")
    prep = sub.add_parser("prepare")
    prep.add_argument("snapshot", type=Path)
    prep.add_argument("packet_labels", type=Path)
    prep.add_argument("output", type=Path)
    merge = sub.add_parser("reconcile")
    merge.add_argument("snapshot", type=Path)
    merge.add_argument("reviewer_a", type=Path)
    merge.add_argument("reviewer_b", type=Path)
    merge.add_argument("output", type=Path, help="New or empty output directory")
    merge.add_argument("--adjudicated", type=Path, help="CSV with resume_id, job_id and relevant")
    args = cli.parse_args()
    snapshot = ablation.from_retrieval(json.loads(args.snapshot.read_text(encoding="utf-8")))
    if args.command == "freeze":
        if args.output.exists():
            cli.error("Choose a new output path to preserve existing batches")
        try:
            batch = freeze_batch(snapshot, args.limit, args.versions)
        except ValueError as exc:
            cli.error(str(exc))
        ablation.save_json(args.output, batch)
        print(f"Frozen {args.limit} résumés across {len(batch['runs'])} versions")
        return
    if args.command == "prepare":
        prepare(snapshot, args.packet_labels, args.output)
        return
    if args.reviewer_a.samefile(args.reviewer_b):
        cli.error("Supply two independently completed workbook copies")
    if args.output.exists() and any(args.output.iterdir()):
        cli.error("Choose a new or empty output directory")
    pool = ablation.review_pool(snapshot)
    expected = {(resume, job) for resume, jobs in pool.items() for job in jobs}
    a, b = (read_workbook(path, expected) for path in (args.reviewer_a, args.reviewer_b))
    resolutions = {}
    if args.adjudicated:
        judgments = ablation.read_judgments(args.adjudicated, pool)
        resolutions = {(resume, job): "1" if value else "0"
                       for resume, jobs in judgments.items() for job, value in jobs.items()}
    accepted, pending = combine(a, b, resolutions)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, header, rows in (
        ("consensus.csv", ("resume_id", "job_id", "relevant"), accepted),
        ("pending.csv", ("resume_id", "job_id", "reviewer_a", "reviewer_b", "status", "relevant"), pending),
    ):
        with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
            table = csv.writer(handle)
            table.writerow(header)
            table.writerows(rows)
    judged = ablation.read_judgments(args.output / "consensus.csv", pool)
    ablation.save_json(args.output / "metrics.json", ablation.score(snapshot, judged))
    ablation.save_json(args.output / "agreement.json", agreement(a, b))
    print(f"Accepted {len(accepted) - len(pending)} of {len(accepted)} pairs. {len(pending)} remain open.")


if __name__ == "__main__":
    main()
