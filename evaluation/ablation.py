"""Collect A–E rankings, export a blind review packet, then score reviewed results."""
import argparse
import csv
import json
from math import ceil
import os
from pathlib import Path
import random
from statistics import mean, median, quantiles
from time import perf_counter, time
from unittest.mock import patch

from evaluation import metrics
from evaluation.retrieval import summarize
from hiremind import agent, config, tools

VERSIONS = {
    "A": None,          # FAISS only
    "B": (False, True), # Adapter without Critic
    "C": (True, False), # Adapter without episodic memory
    "D": (True, True),  # Full base model
    "E": (True, True),  # Full adapter
}


def sample_resumes(split: str, limit: int) -> list[dict]:
    ids = set((config.SPLITS_DIR / f"{split}.txt").read_text().splitlines())
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["id"] in ids]
    rows.sort(key=lambda row: row["id"])
    return random.Random(config.SEED).sample(rows, min(limit, len(rows)))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect(args):
    # FAISS and PyTorch both load native threading libraries on macOS.
    # Initialize the same way as the app, before importing the baseline tools.
    import torch

    from evaluation import baselines
    from hiremind import index

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    resumes = sample_resumes(args.split, args.limit)
    if not resumes:
        raise ValueError("The requested split contains no available résumés")
    if set(args.versions) & {"B", "C", "E"} and not args.adapter:
        raise ValueError("Versions B, C and E require --adapter")
    if args.adapter and not (args.adapter / "adapter_config.json").is_file():
        raise ValueError("The supplied adapter directory has no adapter_config.json")
    settings = {"split": args.split, "resume_ids": [row["id"] for row in resumes],
                "adapter": str(args.adapter.resolve()) if args.adapter else None,
                "adapter_roles": args.adapter_roles, "seed": config.SEED}
    if args.output.exists():
        data = json.loads(args.output.read_text(encoding="utf-8"))
        if data["settings"] != settings:
            raise ValueError("Choose a fresh output path when changing the cohort or adapter")
    else:
        data = {"settings": settings, "runs": {}}
    catalogue = tools._jobs()
    baseline_index = index.load() if "A" in args.versions else None
    for version in dict.fromkeys(args.versions):
        runs = data["runs"].setdefault(version, {})
        adapter = settings["adapter"] if version in {"B", "C", "E"} else ""
        with patch.dict(os.environ, {"HIREMIND_ADAPTER_DIR": adapter or "",
                                     "HIREMIND_ADAPTER_ROLES": args.adapter_roles}):
            for row in resumes:
                if row["id"] in runs:
                    continue
                if args.stop_at and time() >= args.stop_at:
                    print("Reached the run deadline. Saved rankings can be resumed.", flush=True)
                    return
                print(f"{version}: starting case {len(runs) + 1}/{len(resumes)} (résumé {row['id']})", flush=True)
                started = perf_counter()
                try:
                    if version == "A":
                        ids = baselines.faiss_only(row["text"], index=baseline_index)
                        rounds, retries = [], 0
                    else:
                        use_critic, use_episodic = VERSIONS[version]
                        result = agent.run(row["text"], use_critic=use_critic, use_episodic=use_episodic)
                        ids = [item.job_id for item in result.top_jobs]
                        rounds, retries = result.round_rankings, result.retries
                    run = {"job_ids": ids, "round_rankings": rounds, "retries": retries,
                           "category_matches": sum(catalogue[job_id].category == row["category"] for job_id in ids)}
                except ValueError as exc:
                    run = {"error": type(exc).__name__, "job_ids": [], "round_rankings": [], "retries": 0}
                run["seconds"] = round(perf_counter() - started, 3)
                run["device_setting"] = os.environ.get("HIREMIND_DEVICE", "auto") if version != "A" else "cpu"
                runs[row["id"]] = run
                save_json(args.output, data)
                print(f"{version}: {len(runs)}/{len(resumes)} completed", flush=True)


def from_retrieval(data: dict) -> dict:
    """Read historic retrieval snapshots through the same review pipeline."""
    if isinstance(data["runs"], dict):
        return data
    runs = {name: {} for name in ("dense", "shortlist")}
    for row in data["runs"]:
        for name in runs:
            runs[name][row["resume_id"]] = row[name]
    return {"settings": {"resume_ids": [row["resume_id"] for row in data["runs"]]}, "runs": runs}


def top_jobs(job_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(job_ids))[:config.TOP_K]


def review_pool(data: dict) -> dict[str, set[str]]:
    pairs = {resume_id: set() for resume_id in data["settings"]["resume_ids"]}
    for runs in data["runs"].values():
        for resume_id, run in runs.items():
            pairs[resume_id].update(top_jobs(run["job_ids"]))
            if run.get("round_rankings"):
                pairs[resume_id].update(top_jobs(run["round_rankings"][0]))
    return pairs


def make_packet(data: dict, directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Choose a new or empty packet directory to preserve existing judgments")
    directory.mkdir(parents=True, exist_ok=True)
    pool = review_pool(data)
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        resumes = {r["id"]: r["text"] for r in csv.DictReader(handle) if r["id"] in pool}
    rng = random.Random(config.SEED)
    with (directory / "labels.csv").open("w", encoding="utf-8", newline="") as labels:
        table = csv.writer(labels)
        table.writerow(["resume_id", "job_id", "relevant"])
        for resume_id, job_ids in pool.items():
            if not job_ids:
                continue
            shuffled = sorted(job_ids)
            rng.shuffle(shuffled)
            sections = [f"# Résumé {resume_id}", resumes[resume_id]]
            for job_id in shuffled:
                job = tools.get_job_details(job_id)
                table.writerow([resume_id, job_id, ""])
                sections += [f"## Job {job_id} — {job.title}", job.description]
            (directory / f"resume-{resume_id}.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    (directory / "README.md").write_text(
        "# Blind relevance review\n\nUse the posting and résumé evidence for field, core skills and seniority. "
        "Enter 1 for a credible match or 0 for an unsuitable match in labels.csv. Leave genuinely uncertain pairs blank. "
        "Resolve reviewer disagreements before using a merged judgment file.\n", encoding="utf-8"
    )
    print(f"Exported {sum(map(len, pool.values()))} blinded pairs for {sum(bool(jobs) for jobs in pool.values())} résumés")


def read_judgments(path: Path, pool: dict[str, set[str]]) -> dict[str, dict[str, bool]]:
    judged = {resume_id: {} for resume_id in pool}
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["resume_id"], row["job_id"]
            answer = row["relevant"].strip()
            if key[1] not in pool.get(key[0], set()) or key in seen or answer not in {"", "0", "1"}:
                raise ValueError(f"Unknown pair, duplicate, or invalid judgment: {key}")
            seen.add(key)
            if answer:
                judged[key[0]][key[1]] = answer == "1"
    return judged


def fully_judged_ids(data: dict, judgments: dict[str, dict[str, bool]]) -> list[str]:
    runs_by_version = data["runs"]
    return [resume_id for resume_id, jobs in review_pool(data).items()
            if runs_by_version and jobs <= judgments.get(resume_id, {}).keys()
            and all(resume_id in runs and "error" not in runs[resume_id] for runs in runs_by_version.values())]


def score(data: dict, judgments: dict[str, dict[str, bool]]) -> dict:
    runs_by_version = data["runs"]
    complete = fully_judged_ids(data, judgments)
    result = {}
    for name, runs in runs_by_version.items():
        ranked = {resume_id: top_jobs(run["job_ids"]) for resume_id, run in runs.items()}
        result[name] = summarize([(ids, judgments.get(resume_id, {})) for resume_id, ids in ranked.items()], config.TOP_K)
        result[name].update({"completely_judged_resumes": len(complete),
                             "errors": sum("error" in run for run in runs.values()),
                             "empty_slots": sum(config.TOP_K - len(set(ids)) for ids in ranked.values()),
                             "mean_seconds": mean(run["seconds"] for run in runs.values()) if runs else None})
        if not complete:
            continue
        relevant = {resume_id: {job for job, verdict in judgments.get(resume_id, {}).items() if verdict}
                    for resume_id in complete}
        result[name]["precision_at_5"] = mean(metrics.precision_at_k(ranked[i], relevant[i]) for i in complete)
        result[name]["mrr_at_5"] = mean(metrics.mrr(ranked[i], relevant[i]) for i in complete)
        result[name]["pooled_ndcg_at_5"] = mean(metrics.ndcg_at_k(ranked[i], relevant[i]) for i in complete)
        retried = [i for i in complete if len(runs[i].get("round_rankings", [])) > 1]
        result[name]["judged_retries"] = len(retried)
        result[name]["retries_improving_precision"] = sum(
            metrics.precision_at_k(ranked[i], relevant[i]) >
            metrics.precision_at_k(runs[i]["round_rankings"][0], relevant[i]) for i in retried
        )
    return result


def compare_precision(data: dict, judgments: dict[str, dict[str, bool]], baseline: str) -> dict:
    """Calculate paired P@5 changes and a résumé-level bootstrap interval."""
    runs = data["runs"]
    if baseline not in runs:
        raise ValueError(f"Baseline {baseline} is absent from the snapshot")
    complete = fully_judged_ids(data, judgments)
    precisions = {variant: [] for variant in runs}
    for resume_id in complete:
        relevant = {job for job, verdict in judgments.get(resume_id, {}).items() if verdict}
        for variant, cases in runs.items():
            ranked = cases[resume_id]["job_ids"]
            precisions[variant].append(metrics.precision_at_k(ranked, relevant, config.TOP_K))

    comparisons = {}
    for variant, scores in precisions.items():
        if variant == baseline:
            continue
        deltas = [after - before for before, after in zip(precisions[baseline], scores)]
        interval = None
        if len(deltas) >= 2:
            rng = random.Random(config.SEED)
            draws = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000)]
            percentiles = quantiles(draws, n=40)
            interval = [round(percentiles[0], 4), round(percentiles[-1], 4)]
        comparisons[variant] = {
            "precision_at_5": round(mean(scores), 4) if scores else None,
            "delta_precision_at_5": round(mean(deltas), 4) if deltas else None,
            "bootstrap_95_interval": interval,
            "improved": sum(delta > 0 for delta in deltas),
            "unchanged": deltas.count(0),
            "worse": sum(delta < 0 for delta in deltas),
        }
    return {"baseline": baseline, "baseline_precision_at_5": round(mean(precisions[baseline]), 4) if complete else None,
            "fully_judged_resumes": len(complete), "comparisons": comparisons}


def status(data: dict) -> dict:
    total = len(data["settings"]["resume_ids"])
    return {version: {"processed": len(runs), "cohort": total,
                      "errors": sum("error" in row for row in runs.values()),
                      "median_seconds": round(median(row["seconds"] for row in runs.values()), 3) if runs else None}
            for version, runs in data["runs"].items()}


def diagnostics(data: dict) -> dict:
    """Summarize result coverage, latency and category agreement before review."""
    resume_ids = data["settings"]["resume_ids"]
    by_version = data["runs"]
    versions = {}
    successful = {}
    for name in VERSIONS:
        recorded = by_version.get(name, {})
        runs = {resume_id: row for resume_id, row in recorded.items() if "error" not in row}
        successful[name] = runs
        times = sorted(row["seconds"] for row in runs.values())
        counts = [len(top_jobs(row["job_ids"])) for row in runs.values()]
        category_matches = [row["category_matches"] for row in runs.values() if "category_matches" in row]
        versions[name] = {
            "processed": len(recorded), "pending": len(resume_ids) - len(recorded),
            "errors": len(recorded) - len(runs), "top_five_slots_filled": sum(counts),
            "resumes_with_five_jobs": counts.count(config.TOP_K), "resumes_with_no_jobs": counts.count(0),
            "resumes_with_retries": sum(row.get("retries", 0) > 0 for row in runs.values()),
            "median_seconds": round(median(times), 3) if times else None,
            "p90_seconds": times[ceil(0.9 * len(times)) - 1] if times else None,
            "same_category_slots": sum(category_matches) if category_matches else None,
            "category_cohort": len(category_matches),
        }
    comparisons = {}
    for baseline, variant in (("A", "D"), ("B", "E"), ("C", "E"), ("D", "E")):
        left, right = successful[baseline], successful[variant]
        common = left.keys() & right.keys()
        category_deltas = [right[i]["category_matches"] - left[i]["category_matches"]
                           for i in common if "category_matches" in left[i] and "category_matches" in right[i]]
        comparisons[f"{baseline}–{variant}"] = {
            "shared_successful_resumes": len(common),
            "identical_rankings": sum(top_jobs(left[i]["job_ids"]) == top_jobs(right[i]["job_ids"]) for i in common),
            "category_cohort": len(category_deltas),
            "same_category_slots_delta": sum(category_deltas) if category_deltas else None,
        }
    return {"split": data["settings"].get("split"), "cohort": len(resume_ids),
            "planned_cases": len(resume_ids) * len(VERSIONS), "variants": versions, "comparisons": comparisons}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("collect")
    runner.add_argument("--split", choices=("val", "test"), default="val")
    runner.add_argument("--limit", type=int, default=100)
    runner.add_argument("--versions", nargs="+", choices=VERSIONS, default=["A", "D"])
    runner.add_argument("--adapter", type=Path)
    runner.add_argument("--adapter-roles", default="parser,searcher,critic")
    runner.add_argument("--stop-at", type=float, help="Unix timestamp at which to stop between cases")
    runner.add_argument("--output", type=Path, required=True)
    for name in ("packet", "score", "compare"):
        sub = commands.add_parser(name)
        sub.add_argument("snapshot", type=Path)
        sub.add_argument("target", type=Path)
        if name == "compare":
            sub.add_argument("--baseline", default="A")
    commands.add_parser("status").add_argument("snapshot", type=Path)
    commands.add_parser("diagnostics", help="Inspect coverage and latency from an A–E snapshot").add_argument(
        "snapshot", type=Path)
    args = cli.parse_args()
    if args.command == "collect":
        if args.limit < 1:
            cli.error("--limit must be positive")
        collect(args)
    else:
        data = from_retrieval(json.loads(args.snapshot.read_text(encoding="utf-8")))
        if args.command == "packet":
            make_packet(data, args.target)
        elif args.command == "status":
            print(json.dumps(status(data), indent=2))
        elif args.command == "diagnostics":
            if set(data["runs"]) - VERSIONS.keys():
                cli.error("Diagnostics require an A–E study snapshot")
            print(json.dumps(diagnostics(data), indent=2, ensure_ascii=False))
        elif args.command == "compare":
            if args.baseline not in data["runs"]:
                cli.error(f"Baseline {args.baseline} is absent from the snapshot")
            judgments = read_judgments(args.target, review_pool(data))
            print(json.dumps(compare_precision(data, judgments, args.baseline), indent=2))
        else:
            print(json.dumps(score(data, read_judgments(args.target, review_pool(data))), indent=2))


if __name__ == "__main__":
    main()
