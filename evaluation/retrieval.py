"""Compare retrieval on the validation pairs without counting unjudged jobs as misses."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from hiremind import config, tools
from hiremind.memory import Memory
from hiremind.roles import parser, searcher


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reviewed_pairs(pairs: list[dict], labels: list[dict]) -> dict[str, dict[str, bool]]:
    """Keep pairs with all assigned reviews present and in agreement."""
    reviews = defaultdict(dict)
    for row in labels:
        if row["relevant"] in {"0", "1"} and "ai draft" not in (row.get("note") or "").casefold():
            reviews[row["pair_id"]][row["labeller"]] = row["relevant"] == "1"
    result = defaultdict(dict)
    for pair in pairs:
        votes = reviews[pair["pair_id"]]
        if votes.keys() == set(pair["labellers"].split("|")) and len(set(votes.values())) == 1:
            result[pair["resume_id"]][pair["job_id"]] = next(iter(votes.values()))
    return result


def summarize(samples: list[tuple[list[str], dict[str, bool]]], k: int) -> dict:
    """Expose judgment coverage and precision bounds for a sparsely judged pool."""
    positive = negative = unjudged = known = 0
    for ranked, labels in samples:
        shortlisted = list(dict.fromkeys(ranked))[:k]
        positive += sum(labels.get(job_id) is True for job_id in shortlisted)
        negative += sum(labels.get(job_id) is False for job_id in shortlisted)
        unjudged += sum(job_id not in labels for job_id in shortlisted)
        known += sum(labels.values())
    slots = len(samples) * k
    has_labels = any(labels for _, labels in samples)
    return {
        "resumes": len(samples),
        "slots": slots,
        "confirmed_relevant": positive,
        "confirmed_irrelevant": negative,
        "unjudged": unjudged,
        "judgment_coverage": (positive + negative) / slots if slots else None,
        "precision_bounds": [positive / slots, (positive + unjudged) / slots] if slots and has_labels else None,
        "share_of_known_positives_retrieved": positive / known if known else None,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--limit", type=int, help="Maximum number of validation resumes")
    cli.add_argument("--resume-id", action="append", help="Compare only these validation IDs")
    cli.add_argument("--output", type=Path, default=config.ROOT / "outputs" / "retrieval_validation.json")
    args = cli.parse_args()
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    label_dir = config.ROOT / "evaluation" / "labels"
    pairs = read_rows(label_dir / "relevance_pairs.csv")
    label_path = label_dir / "relevance_val.csv"
    reviewed = reviewed_pairs(pairs, read_rows(label_path) if label_path.is_file() else [])
    validation_ids = set((config.SPLITS_DIR / "val.txt").read_text().splitlines())
    ids = {row["resume_id"] for row in pairs} & validation_ids
    if args.resume_id:
        if set(args.resume_id) - ids:
            cli.error("All resume IDs must appear in the validation manifest")
        ids &= set(args.resume_id)
    ids = sorted(ids)[:args.limit]
    resumes = {row["id"]: row for row in read_rows(config.PROCESSED_DIR / "resumes.csv") if row["id"] in ids}
    records = []
    results = {"dense": [], "shortlist": []}
    failures = []
    for resume_id in ids:
        try:
            profile = parser.parse(resumes[resume_id]["text"])
        except (KeyError, ValueError) as exc:
            failures.append({"resume_id": resume_id, "cause": type(exc).__name__})
            continue
        try:
            action, arguments = searcher.next_action(Memory(semantic=profile))
            query = arguments["query"] if action == "search_jobs" else ""
        except ValueError:
            query = ""
        used_fallback = not query
        if used_fallback:
            query = searcher.profile_queries(profile)[0]
        record = {"resume_id": resume_id, "fallback_query": used_fallback}
        for name, retrieve in (
            ("dense", lambda: tools.search_jobs(query, k=config.REVIEW_LIMIT)),
            ("shortlist", lambda: tools.shortlist_jobs(query, profile)),
        ):
            started = perf_counter()
            jobs = retrieve()
            ranked = [job.job_id for job in jobs]
            results[name].append((ranked, reviewed.get(resume_id, {})))
            record[name] = {
                "job_ids": ranked,
                "seconds": round(perf_counter() - started, 4),
                "category_matches_at_5": sum(job.category == resumes[resume_id]["category"] for job in jobs[:5]),
            }
        records.append(record)
        print(f"Compared {len(records)} of {len(ids)} validation resumes", flush=True)
    summary = {
        name: {"at_5": summarize(samples, config.TOP_K), "at_review_limit": summarize(samples, config.REVIEW_LIMIT)}
        for name, samples in results.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "failures": failures, "runs": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
