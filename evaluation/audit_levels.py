"""Measure how much explicit job-title evidence can fill missing level metadata."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from hiremind import tools

TITLE_SIGNALS = {
    "internship": re.compile(r"\b(?:intern|internship|co-op)\b", re.I),
    "entry level": re.compile(r"\b(?:junior|jr\.?|entry[- ]level|new grad(?:uate)?)\b", re.I),
    "mid-senior level": re.compile(r"\b(?:senior|sr\.?|principal)\b", re.I),
    "director": re.compile(r"\bdirector\b", re.I),
    "executive": re.compile(r"\b(?:CEO|CFO|COO|CTO|chief\s+\w+(?:\s+\w+)?\s+officer)\b", re.I),
}


def analyze(jobs):
    total = missing = known_comparable = agree = conflicting_titles = 0
    missing_with_signal = Counter()
    by_signal = {label: Counter() for label in TITLE_SIGNALS}
    for job in jobs:
        total += 1
        level = (job.level or "").strip().casefold()
        signals = [name for name, pattern in TITLE_SIGNALS.items() if pattern.search(job.title)]
        missing += int(not level)
        if len(signals) > 1:
            conflicting_titles += 1
        if len(signals) != 1:
            continue
        signal = signals[0]
        if not level:
            missing_with_signal[signal] += 1
        else:
            known_comparable += 1
            agrees = level == signal
            agree += agrees
            by_signal[signal]["labelled_titles"] += 1
            by_signal[signal]["agrees"] += agrees
    return {
        "total_jobs": total,
        "missing_level": missing,
        "missing_with_one_title_signal": sum(missing_with_signal.values()),
        "missing_by_title_signal": dict(missing_with_signal),
        "labelled_with_one_title_signal": known_comparable,
        "agree_with_provided_level": agree,
        "title_signal_agreement": round(agree / known_comparable, 4) if known_comparable else None,
        "titles_with_conflicting_signals": conflicting_titles,
        "labelled_counts_by_title_signal": {key: dict(values) for key, values in by_signal.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(analyze(tools._jobs().values()), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
