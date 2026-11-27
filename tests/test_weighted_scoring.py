import pytest
from hiremind import agent, matching, tools
from hiremind.roles import critic, parser, searcher
from hiremind.schemas import Job, Preferences, Profile, Judgement


@pytest.fixture(autouse=True)
def no_catalogue_fallback(monkeypatch):
    # These tests supply their own Searcher results.
    monkeypatch.setattr(tools, "shortlist_jobs", lambda *args: [])
    monkeypatch.setattr(tools, "role_shortlist", lambda *args: [])


def test_weights_and_partial_information():
    p = Profile("IT", years_experience=3)
    job = Job("1", "Analyst", "Requirements: 4 years of experience", "IT", location="Boston, MA")
    result = matching.score_weighted(p, job, 8, Preferences(location="Boston"))
    assert result.breakdown.requirements == 8
    assert result.breakdown.experience == 8
    assert result.breakdown.preferences == 10
    assert result.score == 8.2
    assert result.breakdown.coverage == 1
    assert result.breakdown.qualification == 8


def test_unknown_salary_and_years_are_not_scored_as_zero():
    p = Profile("IT")
    job = Job("1", "Analyst", "", "IT", location="Boston, MA")
    result = matching.score_weighted(p, job, 8, Preferences(location="Boston", min_salary=100000))
    assert result.breakdown.experience is None
    assert result.breakdown.preference_coverage == .5
    assert result.breakdown.coverage == .65
    assert result.score == pytest.approx(8.2)
    assert "Salary preference is unscored" in result.reason
    empty = matching.score_weighted(p, job, 8)
    assert empty.score == 8 and empty.breakdown.coverage == .6
    assert empty.breakdown.preferences is None
    benefits = matching.score_weighted(p, job, 8, Preferences(benefits="health insurance"))
    assert benefits.breakdown.preferences is None
    assert "Benefits preference is unscored" in benefits.reason


def test_in_depth_calls_distinct_focused_reviews_and_uses_scores(monkeypatch):
    profile = Profile("IT", years_experience=3)
    job = Job("1", "Analyst", "Requirements: 4 years of experience", "IT", location="Boston, MA")
    monkeypatch.setattr(parser, "parse", lambda _: profile)
    monkeypatch.setattr(searcher, "search", lambda _: [job])
    calls = []
    def focused(profile, job, dimension, preferences):
        calls.append(dimension)
        if dimension == "preferences":
            assert preferences.location == "Boston"
            assert preferences.min_salary is None and preferences.benefits is None
        return {"requirements": 8, "experience": 7, "preferences": 9}[dimension]
    monkeypatch.setattr(critic, "focused", focused)
    result = agent.run("resume", in_depth=True, preferences=Preferences(location="Boston", min_salary=100000, benefits="insurance"))
    assert calls == ["requirements", "experience", "preferences"]
    item = result.top_jobs[0]
    assert item.score == 7.8
    assert item.breakdown.experience == 7 and item.breakdown.preferences == 10
    assert "model preference review: 9/10" in item.reason
    assert item.breakdown.mode == "in-depth"


def test_in_depth_skips_unknown_components_and_uses_fallback(monkeypatch):
    monkeypatch.setattr(parser, "parse", lambda _: Profile("IT"))
    monkeypatch.setattr(searcher, "search", lambda _: [Job("1", "Analyst", "", "IT")])
    calls = []
    monkeypatch.setattr(critic, "focused", lambda p, j, d, prefs: calls.append(d) or None)
    monkeypatch.setattr(critic, "judge", lambda p, jobs: [Judgement("1", 7, "ok")])
    result = agent.run("resume", in_depth=True, preferences=Preferences(min_salary=100000))
    assert calls == ["requirements"]
    assert result.top_jobs[0].breakdown.experience is None
    assert result.top_jobs[0].breakdown.preferences is None
    assert "skipped" in result.top_jobs[0].reason


def test_preference_cannot_promote_below_qualification_cutoff(monkeypatch):
    monkeypatch.setattr(parser, "parse", lambda _: Profile("IT"))
    monkeypatch.setattr(searcher, "search", lambda _: [Job("1", "Analyst", "", "IT", location="Boston")])
    monkeypatch.setattr(critic, "judge", lambda p, jobs: [Judgement("1", 5.8, "ok")])
    score = matching.score_weighted(Profile("IT"), Job("1", "Analyst", "", "IT", location="Boston"),
                                    5.8, Preferences(location="Boston"))
    assert score.score > 6 and score.breakdown.qualification < 6
    assert not agent.run("resume", preferences=Preferences(location="Boston")).top_jobs


def test_experience_cannot_promote_unmatched_network_role(monkeypatch):
    profile = Profile("IT", skills=["Python"], roles=["Software Engineer"], years_experience=8)
    job = Job("1", "Junior Network Engineer", "Requirements: 2 years of experience", "IT", ["Python"])
    item = matching.score_weighted(profile, job, 9)
    assert item.breakdown.requirements == 4.9
    assert item.breakdown.experience == 10
    assert item.breakdown.qualification > 6
    monkeypatch.setattr(parser, "parse", lambda _: profile)
    monkeypatch.setattr(searcher, "search", lambda _: [job])
    monkeypatch.setattr(critic, "judge", lambda p, jobs: [Judgement("1", 9, "ok")])
    assert not agent.run("resume").top_jobs


def test_leadership_seniority_is_explicit_when_year_requirement_unknown():
    p = Profile("FINANCE", years_experience=3, roles=["Data Analyst"])
    j = Job("1", "Finance Directore", "", "FINANCE")
    result = matching.score_weighted(p, j, 8)
    assert result.breakdown.experience == 4
    assert "Seniority estimate" in result.reason


def test_early_career_senior_role_excluded_even_when_model_scores_it_high(monkeypatch):
    p = Profile("IT", skills=["Python"], roles=["Junior Developer"], early_career=True)
    senior = Job("1", "Senior Backend AI Engineer", "Senior role requiring advanced AI experience.", "IT", ["Python"])
    monkeypatch.setattr(parser, "parse", lambda _: p)
    monkeypatch.setattr(searcher, "search", lambda _: [senior])
    monkeypatch.setattr(critic, "focused", lambda *args: 10)
    assert not agent.run("resume", in_depth=True).top_jobs
    junior = Job("2", "Senior Software Engineer", "We are looking for Junior Software Engineers.", "IT")
    assert not matching.early_seniority_conflict(p, junior)


def test_retries_when_fewer_than_five_qualified_jobs(monkeypatch):
    p = Profile("IT")
    monkeypatch.setattr(parser, "parse", lambda _: p)
    seen = []
    def search(memory):
        seen.append(1)
        return [Job(str(len(seen)), "Developer", "", "IT")]
    monkeypatch.setattr(searcher, "search", search)
    monkeypatch.setattr(critic, "judge", lambda p, jobs: [Judgement(jobs[0].job_id, 8, "ok")])
    result = agent.run("resume")
    assert len(result.top_jobs) == 3
    assert result.retries == 2


def test_explicit_core_platform_is_not_overridden_by_generic_skill_overlap():
    p = Profile("IT", skills=["HTML", "CSS", "JavaScript"], early_career=True)
    wix = Job("1", "Web Designer", "Requirements: Proven experience with Wix Studio is required.", "IT", ["HTML", "CSS"])
    item = matching.score_weighted(p, wix, 9)
    assert item.breakdown.requirements < 6
    assert "Wix Studio" in item.reason
    optional = Job("2", "Web Developer", "React experience is preferred.", "IT", ["HTML"])
    assert matching.score_weighted(p, optional, 8).breakdown.requirements >= 6


def test_core_tool_does_not_disappear_because_later_sentence_is_preferred():
    p = Profile("IT", skills=["HTML", "CSS"], early_career=True)
    job = Job("1", "UI Designer", "Experience on Node JS is requiredUnderstanding of the entire process is preferred.", "IT")
    assert "Node.js" in matching.missing_core_tools(p, job)


def test_explicit_posting_title_can_reveal_required_framework():
    p = Profile("IT", skills=["HTML", "JavaScript"], early_career=True)
    job = Job("1", "UI Developer", "Job Title: UI React Location: Remote. Need 8+ Years.", "IT")
    assert "React" in matching.missing_core_tools(p, job)


def test_seniority_in_description_and_explicit_swift_requirement():
    p = Profile("IT", skills=["HTML", "JavaScript"], early_career=True)
    senior = Job("1", "Full Stack Engineer", "The company is seeking a Senior Full Stack Developer.", "IT")
    ios = Job("2", "IOS Developer", "Well versed in Swift and Cocoa touch.", "IT")
    assert matching.early_seniority_conflict(p, senior)
    assert "Swift" in matching.missing_core_tools(p, ios)


def test_specialist_job_titles_require_evidence_of_specialism():
    p = Profile("IT", skills=["Python", "JavaScript"], early_career=True)
    splunk = Job("1", "Splunk Developer", "Python scripting is useful.", "IT")
    ai = Job("2", "Python Developer with AI/ML Skills", "Python is used.", "IT")
    assert "Splunk" in matching.missing_core_tools(p, splunk)
    assert "AI/ML" in matching.missing_core_tools(p, ai)


def test_java_is_not_javascript_and_staff_engineer_is_senior():
    p = Profile("IT", skills=["JavaScript", "Python"], early_career=True)
    java = Job("1", "Consulting Engineer - Java", "", "IT")
    staff = Job("2", "Staff Software Engineer", "", "IT")
    assert "Java" in matching.missing_core_tools(p, java)
    assert matching.early_seniority_conflict(p, staff)
    assert not matching.early_seniority_conflict(p, Job("3", "Staff Accountant", "", "ACCOUNTANT"))


def test_early_career_unknown_years_does_not_read_as_satisfying_year_requirement():
    p = Profile("IT", early_career=True)
    j = Job("1", "Developer", "At least three years of industry experience", "IT")
    result = matching.score_weighted(p, j, 7)
    assert result.breakdown.qualification < 6
    assert "actual total years remain unknown" in result.reason


def test_job_requirements_exclude_optional_skills_and_include_multiple_sections():
    posting = (
        "Basic Qualifications\nSQL and two years of experience.\n"
        "Preferred Qualifications:\nFive years of experience with Java.\n"
        "What You'll Bring:\nExcel and data analysis.\nBenefits:\nPaid time off."
    )
    required = matching._requirements(posting)
    assert "SQL" in required and "Excel" in required
    assert "Java" not in required and "Paid time off" not in required
    assert matching._experience_years(required) == 2


def test_title_credentials_keep_unsubstantiated_roles_below_the_cutoff():
    profile = Profile("HEALTHCARE", skills=["Registered nursing"])
    jobs = [
        Job("1", "Licensed Practical Nurse", "", "HEALTHCARE"),
        Job("2", "Travel Nurse RN", "", "HEALTHCARE"),
        Job("3", "RN or LPN", "", "HEALTHCARE"),
    ]
    judgements = [matching.score_weighted(profile, job, 9) for job in jobs]
    assert judgements[0].breakdown.requirements < 6
    assert "LPN/LVN" in judgements[0].reason
    assert [j.breakdown.requirements for j in judgements[1:]] == [9, 9]


def test_general_javascript_does_not_satisfy_an_ext_js_or_salesforce_title():
    profile = Profile("IT", skills=["JavaScript"])
    assert matching.missing_core_tools(profile, Job("1", "EXT JS Developer", "", "IT")) == ["Ext JS"]
    assert matching.missing_core_tools(profile, Job("2", "Salesforce Developer", "", "IT")) == ["Salesforce"]


def test_platform_gate_uses_specific_resume_evidence_and_respects_alternatives():
    profile = Profile("CONSULTANT", skills=["JavaScript", "SAP"],
                      experience_evidence=["Configured PeopleSoft inventory workflows."])
    jobs = [
        Job("1", "WalkMe Consultant", "", "CONSULTANT", ["JavaScript"]),
        Job("2", "SAP PP Functional Consultant", "", "CONSULTANT"),
        Job("3", "PeopleSoft Consultant", "", "CONSULTANT"),
        Job("4", "HRIS Analyst", "Proven experience with Workday or PeopleSoft is required.", "CONSULTANT"),
    ]
    assert matching.score_weighted(profile, jobs[0], 10).breakdown.requirements < 6
    assert matching.missing_core_tools(profile, jobs[1]) == ["SAP PP"]
    assert not matching.missing_core_tools(profile, jobs[2])
    assert not matching.missing_core_tools(profile, jobs[3])


def test_explicit_license_and_enrollment_requirements_can_block_a_high_score():
    profile = Profile("ACCOUNTANT", education=["Bachelor of Arts in Marketing"])
    audit = Job("1", "Audit Supervisor", "Requirements:Team playerCPA LicenseGood communication skills", "ACCOUNTANT")
    internship = Job("2", "Marketing Intern", "Requirements\n· Currently pursuing a B.A. in Marketing", "ACCOUNTANT")
    assert matching.score_weighted(profile, audit, 10).breakdown.requirements < 6
    assert matching.score_weighted(profile, internship, 10).breakdown.requirements < 6
    profile.education.append("Currently pursuing a B.A. in Marketing")
    assert not matching.missing_required_credentials(profile, internship)
    audit.description = "Requirements: CPA license preferred."
    assert not matching.missing_required_credentials(profile, audit)


def test_advertising_cpa_metric_is_not_an_accounting_credential():
    profile = Profile("DIGITAL-MEDIA", skills=["CPA"], roles=["Media Analyst"],
                      experience_evidence=["Optimized CPA and ROI for digital campaigns."])
    audit = Job("1", "Audit Supervisor", "Requirements:Team playerCPA LicenseGood communication skills", "ACCOUNTANT")
    assert matching.missing_required_credentials(profile, audit) == ["CPA"]
    profile.education.append("Certified Public Accountant")
    assert not matching.missing_required_credentials(profile, audit)


def test_high_score_limited_when_no_detected_required_skills_have_evidence():
    job = Job("1", "Reporting Analyst", "What You Need To Succeed:\nTableau and SQL.", "IT")
    profile = Profile("IT", skills=["Word"])
    item = matching.score_weighted(profile, job, 10)
    assert item.breakdown.requirements == 7.9
    assert "none of the detected required skills" in item.reason
    profile.projects.append("Used Tableau and SQL for a sales dashboard.")
    assert matching.score_weighted(profile, job, 10).breakdown.requirements == 10


def test_total_experience_does_not_establish_pediatric_experience():
    profile = Profile("HEALTHCARE", years_experience=5, roles=["Registered Nurse"])
    job = Job("1", "Registered Nurse", "2 Year minimum of pediatric acute care experience", "HEALTHCARE")
    result = matching.score_weighted(profile, job, 8)
    assert result.breakdown.experience is None
    assert "cannot establish time in that specialty" in result.reason


def test_total_experience_does_not_establish_years_with_a_named_tool():
    profile = Profile("IT", years_experience=5, skills=["Tableau"])
    job = Job("1", "Tableau Consultant", "3+ years experience with implementing Tableau solutions.", "IT")
    result = matching.score_weighted(profile, job, 8)
    assert result.breakdown.experience is None
    assert "specific role or skill" in result.reason
