import numpy as np
import pandas as pd
from openpyxl import load_workbook

from scripts.make_relevance_sheet import _sheet_rows, assign, build_pairs, choose_resumes, write_workbook


def make_data():
    resumes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "category": ["CHEF", "CHEF", "HR", "HR", "HR", "SALES"],
        "text": [f"resume {i}" for i in range(1, 7)],
    })
    jobs = pd.DataFrame({
        "job_id": range(100, 112),
        "title": [f"job {i}" for i in range(12)],
        "company": ["c"] * 12,
        "category": ["CHEF"] * 4 + ["HR"] * 4 + ["SALES"] * 4,
        "level": ["Entry level", None] * 6,
        "description": ["text"] * 12,
    })
    return resumes, jobs


def test_choose_resumes_covers_every_category_first():
    resumes, _ = make_data()
    picked = choose_resumes(resumes, 3, np.random.default_rng(0))
    assert len(picked) == 3
    assert set(resumes.set_index("id").loc[picked, "category"]) == {"CHEF", "HR", "SALES"}


def test_build_pairs_mixes_nearest_and_same_category_jobs():
    resumes, jobs = make_data()
    search = lambda text, k: [111, 110, 109][:k]  # always the SALES jobs, nearest first
    pairs = build_pairs(resumes, jobs, [1], search, np.random.default_rng(0))
    assert len(pairs) == 5
    assert pairs["pair_id"].tolist() == ["P001", "P002", "P003", "P004", "P005"]
    assert set(pairs.loc[pairs["source"] == "nearest", "job_id"]) == {109, 110, 111}
    same = pairs.loc[pairs["source"] == "same_category", "job_id"]
    assert len(same) == 2 and set(same) <= {100, 101, 102, 103}  # CHEF jobs, the resume's category


def test_same_category_jobs_never_repeat_the_nearest_ones():
    resumes, jobs = make_data()
    search = lambda text, k: [100, 101, 102][:k]  # CHEF jobs, so only 103 is left
    pairs = build_pairs(resumes, jobs, [1], search, np.random.default_rng(0))
    assert pairs["job_id"].tolist().count(103) == 1
    assert not pairs["job_id"].duplicated().any()


def test_assign_overlap_and_even_split():
    who = assign(list(range(10)), 2, ("a", "b"), np.random.default_rng(0))
    values = list(who.values())
    assert values.count("a|b") == 2
    assert values.count("a") == values.count("b") == 4


def test_workbook_shows_each_resume_once_and_has_a_guide(tmp_path):
    resumes, jobs = make_data()
    pairs = build_pairs(resumes, jobs, [1, 3], lambda text, k: [111, 110, 109][:k], np.random.default_rng(0))
    path = tmp_path / "sheet.xlsx"
    write_workbook(_sheet_rows(pairs, resumes, jobs), path)
    book = load_workbook(path)
    assert book.sheetnames == ["labels", "how to label"]
    resume_cells = [row[1] for row in book["labels"].iter_rows(min_row=2, values_only=True)]
    assert resume_cells.count("resume 1") == 1 and resume_cells.count("resume 3") == 1
    assert resume_cells.count("(same resume as above)") == 8
