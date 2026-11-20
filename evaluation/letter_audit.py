"""Audit the live Writer against fixed fictional profiles and job postings."""
import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from time import perf_counter

from evaluation.ablation import save_json
from hiremind import config
from hiremind.roles import writer
from hiremind.schemas import Job, Profile


@dataclass
class Case:
    name: str
    profile: Profile
    job: Job
    required: tuple[str, ...]
    excluded: tuple[str, ...] = ()


def cases() -> list[Case]:
    analyst = Profile(
        "INFORMATION-TECHNOLOGY", name="Alex Morgan", roles=["Reporting Analyst"],
        skills=["Python", "SQL", "Docker", "Microsoft Excel"],
        experience_evidence=["Built Python reports for the finance team and saved two hours each week",
                             "Wrote SQL reconciliation queries for monthly billing checks",
                             "Used Docker to package releases for an internal service"],
    )
    florist = Profile("ARTS", name="Avery Khan", roles=["Floral Designer"],
                      experience_evidence=["Arranged seasonal bouquets for a neighborhood shop"])
    accounts = Profile("ACCOUNTANT", name="Jamie Chen", skills=["Microsoft Excel", "SQL"],
                       experience_evidence=["Reconciled vendor invoices with Microsoft Excel each month"],
                       education=["Bachelor of Commerce"])
    student = Profile("INFORMATION-TECHNOLOGY", name="Jordan Lee", skills=["SQL"],
                      education=["BSc Information Systems"])
    return [
        Case("reporting", analyst,
             Job("reporting", "Reporting Analyst", "Build Python reports for forecasting.\nMaintain SQL checks for billing data.",
                 "INFORMATION-TECHNOLOGY", ["Python", "SQL"]),
             ("saved two hours each week", "SQL reconciliation queries", "for forecasting"), ("Docker",)),
        Case("deployment", analyst,
             Job("deployment", "Release Engineer", "Deploy Docker containers for weekly software releases.",
                 "INFORMATION-TECHNOLOGY", ["Docker", "Kubernetes"]),
             ("Used Docker to package releases", "weekly software releases"),
             ("Python", "Kubernetes", "saved two hours")),
        Case("unrelated_evidence", florist,
             Job("data", "Data Analyst", "Maintain SQL reconciliation queries for monthly billing.",
                 "INFORMATION-TECHNOLOGY", ["SQL"]),
             ("seasonal bouquets", "monthly billing"),
             ("I would draw on that work to", "I would bring that approach to", "My skills include SQL")),
        Case("numbered_duties", accounts,
             Job("accounts", "Accounts Payable Associate", "1. Review vendor invoices before payments are processed.\n"
                 "2. Prepare reports for the finance manager.", "ACCOUNTANT", ["Microsoft Excel"]),
             ("Reconciled vendor invoices", "before payments are processed"), ("My skills include SQL",)),
        Case("listed_skill_only", student,
             Job("intern", "Database Intern", "Maintain SQL queries for daily testing.",
                 "INFORMATION-TECHNOLOGY", ["SQL", "PostgreSQL"]),
             ("SQL", "BSc Information Systems", "daily testing"), ("PostgreSQL", "years of experience")),
        Case("no_posting_details", florist, Job("other", "Design Assistant", "", "ARTS"),
             ("Design Assistant", "seasonal bouquets"), ("I would draw on that work to",)),
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Choose a new output path to preserve previous drafts")

    os.environ.update(HIREMIND_DEVICE="cpu", HIREMIND_ADAPTER_DIR="")
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    records = []
    for case in cases():
        started = perf_counter()
        letter = writer.write_personalized(case.profile, case.job)
        elapsed = perf_counter() - started
        folded = letter.casefold()
        missing = [part for part in case.required if part.casefold() not in folded]
        unexpected = [part for part in case.excluded if part.casefold() in folded]
        signature = letter.startswith(case.profile.name + "\n") and letter.endswith(case.profile.name)
        record = {"case": case.name, "profile": asdict(case.profile), "job": asdict(case.job), "draft": letter,
                  "missing": missing, "unexpected": unexpected, "name_and_signature": signature,
                  "seconds": round(elapsed, 3), "passed": not missing and not unexpected and signature}
        records.append(record)
        save_json(args.output, {"model": config.BASE_SLM, "device": "cpu", "cases": records})
        print(f"{case.name}: {'pass' if record['passed'] else 'needs review'} ({elapsed:.2f} s)", flush=True)
    return int(any(not record["passed"] for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
