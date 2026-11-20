"""Make rule-based training labels for the Parser and Critic.

Run as: python -m scripts.make_labels

These labels come from rules, not real recruiters. They teach the model each
task but are not a perfect answer key. Only training and validation resumes are
used; the test split is never read.

Each example uses the exact prompt the role builds at run time
(parser.build_prompt, critic.build_prompt), so training matches inference.
Outputs JSONL files in data/processed/training/:
  parser_train.jsonl, parser_val.jsonl, critic_train.jsonl, critic_val.jsonl
Each line: {"task", "prompt", "completion", "resume_id", ...}. The completion
is the JSON answer the model should give.

The Writer is a fixed template at present, so it has no training examples.
"""
import json
import re

import numpy as np
import pandas as pd

from hiremind import config
from hiremind.roles import critic, parser
from hiremind.schemas import Job, Profile
from hiremind.skills import extract_skills

OUT_DIR = config.PROCESSED_DIR / "training"

# Critic pairs per resume: nearest jobs from the index (hard cases), random jobs
# from the same category, and random jobs from other categories.
PAIRS = {"train": (4, 4, 4), "val": (2, 2, 2)}

_DATE = r"(?:\d{2}/\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{4})"
_RANGE = _DATE + r"\s*(?:to|-|–)\s*(?:" + _DATE + r"|Current|Present)"
_TITLE = r"[A-Z][A-Za-z&/,'\- ]{1,60}?"
_ROLE_AFTER = re.compile(_RANGE + r"\s+(" + _TITLE + r")\s*,?\s+Company Name", re.I)
_ROLE_BEFORE = re.compile(r"(?:^|(?<=[.\n]) ?)(" + _TITLE + r")\s*,?\s+" + _RANGE + r"\s+Company Name", re.M)
_DEGREE = re.compile(
    r"\b(?:Bachelor|Master|Associate|Doctor)(?:'s)? of [A-Z][a-z]+(?: (?:and |in )?[A-Z][a-z]+){0,3}"
    r"(?:\s*:\s*[A-Z][a-z]+(?: (?:and |& )?[A-Z][a-z]+){0,4})?"  # title case, so an all-caps school name is not included
    r"|\bHigh School Diploma\b|\bMBA\b|\bGED\b")
_YEARS = re.compile(r"(?<!\d)(\d{1,2})\s*(?:\+\s*)?years?\s+of\s+(?:[\w&/\-]+\s+){0,3}?experience\b", re.I)


_HEADING_BEFORE = re.compile(r"^(?:(?:Professional|Work|Relevant|Employment)\s+)?(?:Experience|History)\s+", re.I)
_HEADING_AFTER = re.compile(r"\s+(?:Professional\s+|Work\s+)?(?:Experience|History)$", re.I)


def extract_roles(text: str, limit: int = 4) -> list[str]:
    """Job titles written beside a date range, in resume order, as exact phrases.

    Section headings run into titles in this dataset ("Experience Sales Associate"),
    so they are stripped. Sentence-like matches are dropped.
    """
    found = [(m.start(), m.group(1)) for rx in (_ROLE_AFTER, _ROLE_BEFORE) for m in rx.finditer(text)]
    roles = []
    for _, title in sorted(found):
        title = _HEADING_AFTER.sub("", _HEADING_BEFORE.sub("", title.strip(" ,"))).strip(" ,")
        words = title.split()
        if not 1 <= len(words) <= 6 or words[0] in {"I", "My", "We"}:
            continue
        if title.casefold() not in {r.casefold() for r in roles}:
            roles.append(title)
    return roles[:limit]


def extract_education(text: str, limit: int = 2) -> list[str]:
    found = []
    for match in _DEGREE.finditer(text):
        # A graduation month can follow the field of study; it is not part of the degree.
        degree = re.sub(r"(?:\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|\s+(?:Graduated|Expected|Completed|Current|Present))+$",
                        "", match.group(0).strip())
        if degree.casefold() not in {d.casefold() for d in found}:
            found.append(degree)
    return found[:limit]


def extract_years(text: str) -> float | None:
    """Years only when the resume states one number, matching what the Parser accepts at run time."""
    values = {int(v) for v in _YEARS.findall(text)}
    return float(values.pop()) if len(values) == 1 and 0 < next(iter(values)) <= 80 else None


def parser_labels(resume: dict) -> dict:
    """Category from the dataset label; years, education and roles from fixed rules.

    Projects are left empty: LiveCareer resumes have no project sections.
    """
    text = resume["text"]
    return {
        "category": resume["category"],
        "years_experience": extract_years(text),
        "education": extract_education(text),
        "roles": extract_roles(text),
        "projects": [],
    }


def profile_from_labels(labels: dict, text: str) -> Profile:
    return Profile(category=labels["category"], skills=extract_skills(text), years_experience=labels["years_experience"],
                   education=labels["education"], roles=labels["roles"], projects=labels["projects"])


def critic_score(profile: Profile, job: Job) -> tuple[float, str]:
    """Rule score out of 10 from category and skill overlap, with a short reason.

    Years of experience are left out because the Critic prompt says not to score them.
    Same category gives 4 points, a different one 1. Skill overlap (share of the
    job's skills found in the profile) adds up to 6, so a same-category job needs
    at least a third of its skills to reach the cutoff of 6. A job with no
    recognised skills gets 2 neutral points, so unknown is not treated as a mismatch.
    """
    same = profile.category == job.category
    job_skills = list(dict.fromkeys(job.skills))
    have = {s.casefold() for s in profile.skills}
    matched = [s for s in job_skills if s.casefold() in have]
    missing = [s for s in job_skills if s.casefold() not in have]
    if job_skills:
        score = (4 if same else 1) + 6 * len(matched) / len(job_skills)
    else:
        score = (4 if same else 1) + 2
    field = job.category.replace("-", " ").lower()
    parts = [f"Same field ({field})." if same else
             f"Different field: the job is {field}, the profile is {profile.category.replace('-', ' ').lower()}."]
    if not job_skills:
        parts.append("The job lists no recognised skills to compare.")
    else:
        parts.append(f"Matches {len(matched)} of {len(job_skills)} listed skills"
                     + (": " + ", ".join(matched[:5]) + "." if matched else "."))
        if missing:
            parts.append("Missing: " + ", ".join(missing[:5]) + ".")
    return round(score, 1), " ".join(parts)


def choose_jobs(category: str, nearest: list[str], jobs: dict[str, Job], counts: tuple[int, int, int], rng) -> list[tuple[str, str]]:
    """Pick (job_id, source) pairs for one resume without repeats."""
    n_near, n_same, n_other = counts
    picked = [(j, "nearest") for j in nearest[:n_near]]
    used = {j for j, _ in picked}
    same = [j for j, job in jobs.items() if job.category == category and j not in used]
    other = [j for j, job in jobs.items() if job.category != category and j not in used]
    picked += [(str(j), "same_category") for j in rng.choice(same, size=min(n_same, len(same)), replace=False)]
    picked += [(str(j), "other_category") for j in rng.choice(other, size=min(n_other, len(other)), replace=False)]
    return picked


def _write(rows: list[dict], name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} examples")


def main() -> None:
    from hiremind import tools
    from hiremind.index import embed, load

    rng = np.random.default_rng(config.SEED)
    resumes = pd.read_csv(config.PROCESSED_DIR / "resumes.csv")
    jobs = tools._jobs()
    index = load()

    for split, counts in PAIRS.items():
        ids = set(map(int, (config.SPLITS_DIR / f"{split}.txt").read_text().split()))
        part = resumes[resumes["id"].isin(ids)].sort_values("id").to_dict("records")
        vectors = embed([r["text"] for r in part], show_progress=True)

        parser_rows, critic_rows = [], []
        for resume, vector in zip(part, vectors):
            labels = parser_labels(resume)
            parser_rows.append({"task": "parser", "prompt": parser.build_prompt(resume["text"]),
                                "completion": json.dumps(labels, ensure_ascii=False), "resume_id": resume["id"]})
            profile = profile_from_labels(labels, resume["text"])
            nearest = [j for j, _ in index.search_vector(vector, counts[0])]
            for job_id, source in choose_jobs(resume["category"], nearest, jobs, counts, rng):
                job = jobs[job_id]
                score, reason = critic_score(profile, job)
                critic_rows.append({"task": "critic", "prompt": critic.build_prompt(profile, job),
                                    "completion": json.dumps({"score": score, "reason": reason}, ensure_ascii=False),
                                    "resume_id": resume["id"], "job_id": job_id, "source": source})
        _write(parser_rows, f"parser_{split}.jsonl")
        _write(critic_rows, f"critic_{split}.jsonl")


if __name__ == "__main__":
    main()
