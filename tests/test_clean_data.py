import pandas as pd
import pytest

from scripts.clean_data import balance, clean_jobs, clean_resumes, map_job_category, normalize_text
from scripts.split_data import split


@pytest.mark.parametrize("title, category", [
    ("Senior Accountant", "ACCOUNTANT"),
    ("Patient Advocate", "ADVOCATE"),
    ("Automotive Sales Manager", "AUTOMOBILE"),
    ("Sales Consultant", "SALES"),
    ("Leasing Consultant", "SALES"),
    ("Consultant", "CONSULTANT"),
    ("Software Engineer", "INFORMATION-TECHNOLOGY"),
    ("Lead Software Engineer, TIDAL Artist Platform", "INFORMATION-TECHNOLOGY"),
    ("Mechanical Engineer", "ENGINEERING"),
    ("IT Support Specialist", "INFORMATION-TECHNOLOGY"),
    ("HR Generalist", "HR"),
    ("Registered Nurse (RN)", "HEALTHCARE"),
    ("Public Relations Intern", "PUBLIC-RELATIONS"),
])
def test_title_rules(title, category):
    assert map_job_category(title) == category


# Each of these was mapped wrongly by an earlier draft of the rules.
@pytest.mark.parametrize("title, wrong", [
    ("PCU, Registered Nurse (RN) - $46 - $57/hr", "HR"),
    ("Assistant Store Manager ($13-$16/hr)", "HR"),
    ("Subway Sandwich Artist", "ARTS"),
    ("State Farm Agent", "AGRICULTURE"),
    ("Night Auditor - Part Time", "ACCOUNTANT"),
    ("Wind Technician II - Jordan Creek Wind Farm", "AGRICULTURE"),
    ("Business Consultant (PR-1251020:PA)", "PUBLIC-RELATIONS"),
    ("is it for you", "INFORMATION-TECHNOLOGY"),
])
def test_known_false_matches(title, wrong):
    assert map_job_category(title) != wrong


@pytest.mark.parametrize("title", ["Receptionist", "Associate Attorney", "Warehouse Associate"])
def test_roles_with_no_category_are_unmapped(title):
    assert map_job_category(title) is None


def test_apparel_industry_needs_a_retail_title():
    assert map_job_category("Assistant Store Leader", ["Retail Apparel and Fashion"]) == "APPAREL"
    assert map_job_category("Application Architect", ["Retail Apparel and Fashion"]) is None
    assert map_job_category("Assistant Store Leader", ["Banking"]) is None


def test_normalize_text():
    assert normalize_text("  a   b \n\n \t c\xa0d  ") == "a b\nc d"


def test_balance_caps_large_categories_and_keeps_small_ones():
    jobs = pd.DataFrame({"category": ["BIG"] * 100 + ["SMALL"] * 5})
    pool = balance(jobs, size=25, seed=1)
    assert pool["category"].value_counts().to_dict() == {"BIG": 20, "SMALL": 5}


def test_balance_is_repeatable():
    jobs = pd.DataFrame({"category": ["BIG"] * 100 + ["SMALL"] * 5, "n": range(105)})
    assert balance(jobs, 25, seed=1)["n"].tolist() == balance(jobs, 25, seed=1)["n"].tolist()


def test_clean_resumes_drops_short_and_duplicates(tmp_path):
    folder = tmp_path / "resume-dataset" / "Resume"
    folder.mkdir(parents=True)
    long_text = " ".join(f"word{i}" for i in range(60))
    pd.DataFrame({
        "ID": [3, 1, 2, 4],
        "Resume_str": [long_text, long_text.replace(" ", "   "), "too short", long_text + " more"],
        "Resume_html": ["", "", "", ""],
        "Category": ["HR", "HR", "HR", "CHEF"],
    }).to_csv(folder / "Resume.csv", index=False)

    result = clean_resumes(tmp_path)
    assert list(result.columns) == ["id", "category", "text"]
    assert len(result) == 2  # the short one and the whitespace-only duplicate are gone
    assert sorted(result["category"]) == ["CHEF", "HR"]


def test_clean_jobs_end_to_end(tmp_path):
    raw = tmp_path / "linkedin-job-postings"
    (raw / "jobs").mkdir(parents=True)
    (raw / "mappings").mkdir()
    text = " ".join(f"word{i}" for i in range(60))
    pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5],
        "title": ["Line Cook", "Line Cook", "Receptionist", "Store Manager", "Line Cook"],
        "company_name": ["a", "b", "c", "d", "e"],
        "location": ["x"] * 5,
        "description": [text, text + "  ", text + " three", text + " four", "too short"],
        "formatted_experience_level": ["Entry level", None, None, "Director", None],
        "formatted_work_type": ["Full-time"] * 5,
    }).to_csv(raw / "postings.csv", index=False)
    pd.DataFrame({"job_id": [1, 1, 4], "industry_id": [10, 99, 11]}).to_csv(raw / "jobs" / "job_industries.csv", index=False)
    # Industry 99 has no name, like some ids in the real table.
    pd.DataFrame({"industry_id": [10, 11, 99], "industry_name": ["Restaurants", "Retail Apparel and Fashion", None]}).to_csv(raw / "mappings" / "industries.csv", index=False)
    pd.DataFrame({"job_id": [1], "skill_abr": ["MGMT"]}).to_csv(raw / "jobs" / "job_skills.csv", index=False)
    pd.DataFrame({"skill_abr": ["MGMT"], "skill_name": ["Management"]}).to_csv(raw / "mappings" / "skills.csv", index=False)

    jobs = clean_jobs(tmp_path, size=10)
    # 2 is a whitespace-only duplicate of 1, 3 has no category, 5 is too short.
    assert jobs["job_id"].tolist() == [4, 1]
    assert jobs["category"].tolist() == ["APPAREL", "CHEF"]
    cook = jobs[jobs["job_id"] == 1].iloc[0]
    assert cook["industries"] == "Restaurants"
    assert cook["functions"] == "Management"
    assert pd.isna(cook["skills"]) or cook["skills"] == ""  # the fake description names no skills
    assert cook["level"] == "Entry level"


def test_split_has_no_overlap_and_keeps_the_mix(tmp_path):
    resumes = pd.DataFrame({
        "id": range(400),
        "category": ["A"] * 200 + ["B"] * 100 + ["C"] * 100,
    })
    path = tmp_path / "resumes.csv"
    resumes.to_csv(path, index=False)

    first = split(path, tmp_path / "s1", seed=1)
    train, val, test = set(first["train"]), set(first["val"]), set(first["test"])
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(range(400))
    assert (len(train), len(val), len(test)) == (280, 60, 60)
    assert sum(i < 200 for i in test) == 30  # half of the test split is category A
    assert (tmp_path / "s1" / "train.txt").exists()

    assert split(path, tmp_path / "s2", seed=1) == first
