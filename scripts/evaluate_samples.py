"""Run fictional resumes repeatedly and save auditable local results.

Run: python -m scripts.evaluate_samples --runs 5
Outputs are local and git-ignored; never use private resumes without consent.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from hiremind import agent, config, slm, tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=config.ROOT / "outputs" / "sample_evaluation.json")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    samples_dir = config.DATA_DIR / "samples"
    files = sorted(samples_dir.glob("sample_*.txt"))
    example = samples_dir / "test_resume.txt"
    if example.is_file():
        files.append(example)
    if not files:
        parser.error("Add fictional sample_*.txt files or test_resume.txt to data/samples first")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for run_no in range(1, args.runs + 1):
        for path in files:
            text = path.read_text(encoding="utf8")
            start = perf_counter()
            print(f"Starting {path.name} run {run_no}/{args.runs}", flush=True)
            try:
                result = agent.run(text)
                jobs = []
                for judgment in result.top_jobs:
                    job = tools.get_job_details(judgment.job_id)
                    jobs.append({"job_id": job.job_id, "title": job.title, "score": judgment.score,
                                 "reason": judgment.reason, "level": job.level})
                entry = {"file": path.name, "run": run_no, "seconds": round(perf_counter() - start, 2),
                         "profile": asdict(result.profile), "jobs": jobs, "retries": result.retries,
                         "steps": [asdict(s) for s in result.steps],
                         "letter": agent.cover_letter(result.profile, jobs[0]["job_id"]) if jobs else None,
                         "device": str(slm.load()[1].device)}
            except Exception as exc:
                entry = {"file": path.name, "run": run_no, "seconds": round(perf_counter() - start, 2),
                         "error": f"{type(exc).__name__}: {exc}"}
            results.append(entry)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
            print(f"Completed {path.name} run {run_no}: {entry.get('error', str(len(entry.get('jobs', []))) + ' jobs')} in {entry['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
