"""Make the sheets for hand-labelling job relevance on validation resumes.

Run as: python -m scripts.make_relevance_sheet

The labels are the reference the relevance rule's cut-offs are tuned against
(see docs/relevance_labelling.md). Only validation resumes are used.

Outputs:
- evaluation/labels/relevance_pairs.csv: pair ids, resume and job ids, how each
  job was picked and who labels it. Ids only, so it is committed to git.
- data/labels/relevance_<name>.xlsx: one workbook per labeller with the text to
  read. It holds dataset text, so it is shared on Drive, not in git.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from hiremind import config

N_RESUMES = 40
NEAREST = 3  # nearest jobs from the index, a mix of good and borderline matches
SAME_CATEGORY = 2  # random jobs from the resume's category, often a skill or level mismatch
OVERLAP = 6  # resumes labelled by both people, to measure agreement
LABELLERS = ("arslan", "hassaan")
EXCERPT = 2500  # characters of resume and job text shown

KEY_FILE = config.ROOT / "evaluation" / "labels" / "relevance_pairs.csv"
SHEET_DIR = config.DATA_DIR / "labels"

GUIDE = [
    "How to label",
    "",
    "Each row is a resume and a job. Put 1 in the 'relevant' column if this person could reasonably apply",
    "for the job and be a credible candidate, and 0 if not.",
    "",
    "A job is relevant when all three hold:",
    "1. Field: it is the same kind of work, or work this person's experience clearly leads to.",
    "2. Skills: they have most of the main skills the job asks for, or close equivalents.",
    "3. Level: the seniority roughly fits. An intern is not relevant for a director role, and a",
    "   director is not relevant for an entry-level role.",
    "",
    "Judge from the text only, and do not look at the other person's sheet.",
    "Texts are cut to the first 2,500 characters. Judge from what is shown.",
    "Rows from the same resume are grouped. The resume text is shown on the first row of each group.",
    "If you are unsure, pick your best guess and write a short note.",
]


def choose_resumes(resumes: pd.DataFrame, n: int, rng: np.random.Generator) -> list[int]:
    """One resume from every category first, then random ones up to n."""
    ids = [rng.choice(g["id"].to_numpy()) for _, g in resumes.groupby("category")]
    rest = resumes.loc[~resumes["id"].isin(ids), "id"].to_numpy()
    ids += list(rng.choice(rest, size=max(0, n - len(ids)), replace=False))
    return [int(i) for i in ids[:n]]


def build_pairs(resumes: pd.DataFrame, jobs: pd.DataFrame, resume_ids: list[int], search, rng: np.random.Generator) -> pd.DataFrame:
    """`search(text, k)` returns job ids, nearest first."""
    rows = []
    for resume_id in resume_ids:
        resume = resumes.loc[resumes["id"] == resume_id].iloc[0]
        nearest = [int(j) for j in search(resume["text"], NEAREST)]
        pool = jobs.loc[(jobs["category"] == resume["category"]) & ~jobs["job_id"].isin(nearest), "job_id"].to_numpy()
        same = [int(j) for j in rng.choice(pool, size=min(SAME_CATEGORY, len(pool)), replace=False)]
        picked = [(j, "nearest") for j in nearest] + [(j, "same_category") for j in same]
        for k in rng.permutation(len(picked)):  # so the nearest jobs are not always first
            rows.append({"resume_id": resume_id, "job_id": picked[k][0], "source": picked[k][1]})
    pairs = pd.DataFrame(rows)
    pairs.insert(0, "pair_id", [f"P{i:03d}" for i in range(1, len(pairs) + 1)])
    return pairs


def assign(resume_ids: list[int], overlap: int, labellers: tuple[str, ...], rng: np.random.Generator) -> dict[int, str]:
    """Map each resume to who labels it: `overlap` resumes go to everyone, the rest are split evenly."""
    order = [int(i) for i in rng.permutation(resume_ids)]
    who = {r: "|".join(labellers) for r in order[:overlap]}
    for k, r in enumerate(order[overlap:]):
        who[r] = labellers[k % len(labellers)]
    return who


def _sheet_rows(pairs: pd.DataFrame, resumes: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    text = resumes.set_index("id")["text"]
    job = jobs.set_index("job_id")
    rows, last = [], None
    for p in pairs.itertuples():
        j = job.loc[p.job_id]
        rows.append({
            "pair_id": p.pair_id,
            "resume": text[p.resume_id][:EXCERPT] if p.resume_id != last else "(same resume as above)",
            "job_title": j["title"],
            "company": j["company"],
            "job_level": j["level"] if pd.notna(j["level"]) else "",
            "job_description": j["description"][:EXCERPT],
            "relevant": "",
            "note": "",
        })
        last = p.resume_id
    return pd.DataFrame(rows)


def write_workbook(rows: pd.DataFrame, path: Path) -> None:
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="labels", index=False)
        pd.DataFrame({"guide": GUIDE}).to_excel(writer, sheet_name="how to label", index=False, header=False)
        sheet = writer.sheets["labels"]
        widths = {"A": 9, "B": 70, "C": 30, "D": 18, "E": 16, "F": 70, "G": 10, "H": 30}
        for col, width in widths.items():
            sheet.column_dimensions[col].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "C2"
        check = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
        check.add(f"G2:G{len(rows) + 1}")
        sheet.add_data_validation(check)
        writer.sheets["how to label"].column_dimensions["A"].width = 110


def main() -> None:
    from hiremind.index import load

    rng = np.random.default_rng(config.SEED)
    resumes = pd.read_csv(config.PROCESSED_DIR / "resumes.csv")
    val = set(map(int, (config.SPLITS_DIR / "val.txt").read_text().split()))
    resumes = resumes[resumes["id"].isin(val)]
    jobs = pd.read_csv(config.PROCESSED_DIR / "jobs.csv", usecols=["job_id", "title", "company", "category", "level", "description"])
    index = load()

    resume_ids = choose_resumes(resumes, N_RESUMES, rng)
    pairs = build_pairs(resumes, jobs, resume_ids, lambda text, k: [j for j, _ in index.search(text, k)], rng)
    who = assign(resume_ids, OVERLAP, LABELLERS, rng)
    pairs["labellers"] = pairs["resume_id"].map(who)

    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(KEY_FILE, index=False)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for name in LABELLERS:
        mine = pairs[pairs["labellers"].str.split("|").map(lambda names: name in names)]
        write_workbook(_sheet_rows(mine, resumes, jobs), SHEET_DIR / f"relevance_{name}.xlsx")
        print(f"{name}: {len(mine)} pairs from {mine['resume_id'].nunique()} resumes")
    print(f"{len(pairs)} pairs, {pairs['source'].value_counts().to_dict()}, both label {int((pairs['labellers'].str.contains('[|]')).sum())}")


if __name__ == "__main__":
    main()
