import json

import numpy as np
import pytest

from hiremind.roles import critic, parser
from hiremind.schemas import Job, Profile
from scripts.make_labels import (choose_jobs, critic_score, extract_education, extract_roles, extract_years,
                                 parser_labels, profile_from_labels)

RESUME = ("SALES ASSOCIATE Summary Motivated seller with 5 years of retail experience. Experience "
          "12/2016 to Current Sales Associate Company Name － City , State Respond to members. "
          "08/2016 to 11/2016 Marketing Coordinator Company Name － City , State Promote events. "
          "Maintenance & Avionics Coordinator 05/2000 to 08/2005 Company Name City , State Managed parts. "
          "Education May 2016 Bachelor of Science : Communications Public Relations SALEM STATE "
          "Master of Social Work Graduated 2012 "
          "1990 High School Diploma Bishop High School Skills Microsoft Excel, customer service")


def test_roles_are_exact_phrases_in_resume_order():
    roles = extract_roles(RESUME)
    assert roles == ["Sales Associate", "Marketing Coordinator", "Maintenance & Avionics Coordinator"]
    assert all(role in RESUME for role in roles)


def test_section_headings_and_sentences_are_not_roles():
    text = ("Professional Experience Senior HR 01/2010 to 02/2012 Company Name City. "
            "HEALTHCARE CLINICAL ANALYST Professional Experience 03/2012 to Current Company Name City. "
            "I provide constant process improvement 01/2009 to 12/2009 Company Name City")
    assert extract_roles(text) == ["Senior HR", "HEALTHCARE CLINICAL ANALYST"]


def test_education_phrases():
    assert extract_education(RESUME, limit=3) == ["Bachelor of Science : Communications Public Relations", "Master of Social Work", "High School Diploma"]


def test_years_only_when_one_number_is_stated():
    assert extract_years(RESUME) == 5.0
    assert extract_years("I have 3 years of experience and 7 years of sales experience") is None
    assert extract_years("Worked 2010 to 2015") is None


def test_parser_labels_pass_the_runtime_parser_checks():
    labels = parser_labels({"text": RESUME, "category": "SALES"})
    assert labels["category"] == "SALES"
    for name in ("education", "roles"):
        assert all(value.casefold() in RESUME.casefold() for value in labels[name])
    # The runtime parser keeps a years value only if the resume states it.
    assert f"{labels['years_experience']:g} years" in RESUME


def make_job(category="SALES", skills=("Microsoft Excel", "Customer service", "CRM software")):
    return Job(job_id="1", title="Sales Rep", description="Sell things.", category=category, skills=list(skills))


def test_critic_score_rewards_category_and_skills():
    profile = Profile(category="SALES", skills=["Microsoft Excel", "Customer service"])
    same, reason = critic_score(profile, make_job())
    other, _ = critic_score(profile, make_job(category="CHEF"))
    assert same == pytest.approx(4 + 6 * 2 / 3, abs=0.05)
    assert other == pytest.approx(1 + 6 * 2 / 3, abs=0.05)
    assert "Same field" in reason and "Matches 2 of 3" in reason and "Missing: CRM software" in reason


def test_critic_score_stays_between_0_and_10():
    profile = Profile(category="SALES", skills=["Microsoft Excel", "Customer service", "CRM software"])
    assert critic_score(profile, make_job())[0] == 10
    assert critic_score(Profile(category="CHEF"), make_job(category="HR"))[0] == 1


def test_unknown_job_skills_are_neutral_not_a_mismatch():
    score, reason = critic_score(Profile(category="SALES"), make_job(skills=()))
    assert score == 6 and "no recognised skills" in reason


def test_one_skill_in_five_is_below_the_cutoff_even_in_the_same_field():
    job = make_job(skills=("Microsoft Excel", "A", "B", "C", "D"))
    assert critic_score(Profile(category="SALES", skills=["Microsoft Excel"]), job)[0] < 6


def test_choose_jobs_has_no_repeats_and_respects_counts():
    jobs = {str(i): make_job(category="SALES" if i < 10 else "CHEF") for i in range(20)}
    picked = choose_jobs("SALES", ["0", "1"], jobs, (2, 3, 4), np.random.default_rng(0))
    ids = [j for j, _ in picked]
    assert len(ids) == len(set(ids)) == 9
    assert [s for _, s in picked].count("same_category") == 3
    assert all(jobs[j].category == "CHEF" for j, s in picked if s == "other_category")


def test_examples_use_the_runtime_prompts():
    labels = parser_labels({"text": RESUME, "category": "SALES"})
    profile = profile_from_labels(labels, RESUME)
    assert parser.build_prompt(RESUME).startswith("Extract a resume profile as JSON")
    prompt = critic.build_prompt(profile, make_job())
    assert json.dumps(profile.category) in prompt and "Sales Rep" in prompt
