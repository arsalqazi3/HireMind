"""Measure the real matching and cover-letter pipeline on CPU validation cases."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
from time import perf_counter

from evaluation import ablation, retrieval
from hiremind import agent, config
from scripts.benchmark_slm import peak_rss_mib


def benchmark_case(resume: dict, in_depth: bool) -> dict:
    durations = defaultdict(float)
    phase = None
    started = last_transition = perf_counter()
    reviewed = {}

    def progress(event):
        nonlocal phase, last_transition
        current = event["phase"]
        if current != phase:
            now = perf_counter()
            if phase is not None:
                durations[phase] += now - last_transition
            phase, last_transition = current, now
        if current == "scoring":
            reviewed[event["round"]] = event["completed"]

    result = agent.run(resume["text"], progress=progress, in_depth=in_depth)
    matching_seconds = perf_counter() - started
    job_ids = [job.job_id for job in result.top_jobs]
    letter_seconds = None
    if job_ids:
        started = perf_counter()
        letter = agent.cover_letter(result.profile, job_ids[0])
        if not letter.strip():
            raise ValueError("Expected a cover letter for the highest ranked job")
        letter_seconds = round(perf_counter() - started, 3)
    return {
        "resume_id": resume["id"], "job_ids": job_ids, "retries": result.retries,
        "matching_seconds": round(matching_seconds, 3), "cover_letter_seconds": letter_seconds,
        "phase_seconds": {name: round(value, 3) for name, value in durations.items()},
        "reviewed_jobs_by_round": reviewed, "cumulative_process_peak_mib": round(peak_rss_mib(), 1),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--limit", type=int, default=3, help="Number of seeded validation cases")
    cli.add_argument("--resume-id", action="append", help="Benchmark only these validation IDs, in order")
    cli.add_argument("--in-depth", action="store_true")
    args = cli.parse_args()
    if args.limit < 1 or args.output.exists():
        cli.error("Provide a positive case count and a fresh output path")
    if args.resume_id:
        ids = set((config.SPLITS_DIR / "val.txt").read_text(encoding="utf-8").splitlines())
        rows = retrieval.read_rows(config.PROCESSED_DIR / "resumes.csv")
        catalogue = {row["id"]: row for row in rows if row["id"] in ids}
        if len(set(args.resume_id)) != len(args.resume_id) or any(i not in catalogue for i in args.resume_id):
            cli.error("Each requested ID must be unique and available in the validation split")
        resumes = [catalogue[i] for i in args.resume_id]
    else:
        resumes = ablation.sample_resumes("val", args.limit)
    if not resumes:
        cli.error("No validation résumés are available")

    os.environ.update(HIREMIND_DEVICE="cpu", HIREMIND_ADAPTER_DIR="")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    # Initialize PyTorch first to avoid a native OpenMP conflict with FAISS on macOS.
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    output = {"split": "val", "device": "cpu", "in_depth": args.in_depth, "model": config.BASE_SLM,
              "torch_threads": torch.get_num_threads(), "requested_ids": [row["id"] for row in resumes], "runs": []}
    ablation.save_json(args.output, output)
    for index, row in enumerate(resumes, start=1):
        print(f"Starting CPU case {index}/{len(resumes)} (résumé {row['id']})", flush=True)
        try:
            result = benchmark_case(row, args.in_depth)
        except (ValueError, RuntimeError) as exc:
            result = {"resume_id": row["id"], "error": type(exc).__name__}
        output["runs"].append(result)
        ablation.save_json(args.output, output)
        detail = f"{result['matching_seconds']:g} s matching" if "matching_seconds" in result else result["error"]
        print(f"Completed CPU case {index}/{len(resumes)} ({detail})", flush=True)
    return int(any("error" in result for result in output["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
