import csv

import pytest

from hiremind.skills import SKILLS_FILE, all_skills, extract_skills, tokenize


@pytest.mark.parametrize("text, skill", [
    ("Proficient in Microsoft Excel and Word.", "Microsoft Excel"),
    ("MS Word, Excel, Outlook", "Microsoft Excel"),
    ("Strong SQL skills", "SQL"),
    ("C/C++ and C# required", "C++"),
    ("C/C++ and C# required", "C#"),
    ("experience with .NET and Azure", ".NET"),
    ("HTML/CSS", "CSS"),
    ("Node.js services", "Node.js"),
    ("cold-calling experience", "Cold calling"),
    ("cold\ncalling experience", "Cold calling"),
    ("writes LESSON PLANS weekly", "Lesson planning"),
    ("holds an A&P certificate", "A&P license"),
    ("current CPR and BLS", "CPR certification"),
])
def test_finds_skills(text, skill):
    assert skill in extract_skills(text)


# Each of these was matched wrongly by an earlier version of the list.
@pytest.mark.parametrize("text, skill", [
    ("We excel at helping clients", "Microsoft Excel"),
    ("Microsoft Access is available", "Microsoft Excel"),
    ("Access to free benefits", "Microsoft Access"),
    ("Rust-Oleum Corporation is hiring", "Rust"),
    ("Epic Owners and Program Managers", "Epic"),
    ("Apply through your Workday account", "Workday"),
    ("clean the kitchen blenders and mixers", "3D modeling"),
    ("free counseling through our Employee Assistance Program", "Counseling"),
    ("Medical, dental and mental health benefits", "Mental health"),
    ("a comprehensive compensation and benefits package", "Compensation"),
    ("available upon request in connection with the recruitment process", "Recruiting"),
    ("contact Guidehouse Recruiting", "Recruiting"),
    ("Home Mortgage Program for employees", "Mortgage lending"),
    ("Reacts well under pressure", "React"),
])
def test_ignores_false_matches(text, skill):
    assert skill not in extract_skills(text)


def test_result_is_sorted_and_unique():
    found = extract_skills("Python, python and PYTHON with SQL and Excel")
    assert found == sorted(set(found))
    assert found.count("Python") == 1


def test_empty_text():
    assert extract_skills("") == []


def test_tokenize_splits_on_hyphen_and_slash_and_ignores_trailing_dots():
    assert tokenize("cold-calling, HTML/CSS and Excel.") == ["cold", "calling", "HTML", "CSS", "and", "Excel"]


def test_every_row_can_match():
    with open(SKILLS_FILE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert row["terms"] or row["exact_terms"], row["skill"]
        for term in [*filter(None, row["terms"].split("|")), *filter(None, row["exact_terms"].split("|"))]:
            assert row["skill"] in extract_skills(f"has {term} here"), term


def test_skill_names_are_unique():
    names = all_skills()
    assert len(names) == len(set(names))
