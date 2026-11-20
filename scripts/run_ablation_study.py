"""Run each ablation variant in a fresh process with a shared time budget."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
from time import monotonic, time


def collect_variant(command: list[str], version: str, deadline: float) -> int:
    remaining = deadline - monotonic()
    if remaining <= 0:
        return 124
    try:
        return subprocess.run(command, timeout=remaining).returncode
    except subprocess.TimeoutExpired:
        print(f"{version} reached the deadline. Completed cases remain saved.", flush=True)
        return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--hours", type=float, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0 or args.limit < 1:
        parser.error("A finite positive time budget and case count are required")
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("The adapter directory must contain adapter_config.json")
    if args.packet and args.packet.exists() and any(args.packet.iterdir()):
        parser.error("Choose a new packet directory to preserve existing judgments")

    budget = 3600 * args.hours
    deadline = monotonic() + budget
    stop_at = time() + budget
    prefix = [sys.executable, "-m", "evaluation.ablation"]
    print(f"Collecting up to {args.limit} {args.split} résumés per variant", flush=True)
    outcome = 0
    for version in "ABCDE":
        if monotonic() >= deadline:
            break
        command = [*prefix, "collect", "--split", args.split, "--versions", version,
                   "--adapter", str(args.adapter), "--limit", str(args.limit),
                   "--stop-at", str(stop_at), "--output", str(args.output)]
        outcome = collect_variant(command, version, deadline)
        if outcome and outcome != 124 and monotonic() < deadline:
            print(f"{version} exited with {outcome}. Retrying from the checkpoint.", flush=True)
            outcome = collect_variant(command, version, deadline)
        if outcome:
            print(f"{version} could not continue. Existing results remain saved.", flush=True)
            break
    if args.output.is_file():
        subprocess.run([*prefix, "status", str(args.output)], check=True)
        if args.packet:
            subprocess.run([*prefix, "packet", str(args.output), str(args.packet)], check=True)
    return outcome


if __name__ == "__main__":
    raise SystemExit(main())
