"""Conservative, explainable adjustments to model-generated job judgements.

These heuristics are provisional, not a validated measure of job relevance.
"""
import re

from hiremind import config
from hiremind.schemas import Job, Judgement, Profile
from hiremind.skills import extract_skills


_NUMBERS = {word: number for number, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"))}
_NUMBER = r"(?:\d{1,2}|" + "|".join(_NUMBERS) + r")"
_EXPERIENCE = re.compile(
    rf"\b(?P<start>{_NUMBER})(?:\s*(?:-|–|to|or)\s*{_NUMBER})?\s*\+?\s+years?[’']?"
    r"(?:\s+of)?\s+(?P<scope>[^\d.\n]{0,65}?)\bexperience(?:\b|(?-i:(?=[A-Z])))"
    r"(?:[ \t]+(?:with|in|as)[ \t]+(?P<after>[^.;\n*•]{1,65}))?",
    re.I,
)
_CLINICAL_SCOPE = re.compile(r"\b(?:pediatric\w*|neonatal|NICU|ICU|obstetric\w*|critical care|labor and delivery)\b", re.I)
_ROLE_SCOPE = re.compile(r"\b(?:data engineering|software development|software engineering)\b", re.I)


def _experience_mentions(text: str):
    for match in _EXPERIENCE.finditer(text):
        context = text[max(0, match.start() - 45):match.end() + 35]
        if re.search(r"\b(?:preferred|nice to have|bonus)\b|\bis a plus\b", context, re.I):
            continue
        value = match.group("start").lower()
        years = int(value) if value.isdigit() else _NUMBERS[value]
        yield years, (match.group("scope") + " " + (match.group("after") or "")).strip()


def _experience_years(text: str) -> int | None:
    """Find explicit experience durations; unknown remains unknown."""
    return min((years for years, _ in _experience_mentions(text)), default=None)


def experience_scope(job: Job, requirement_text: str) -> str | None:
    """Identify durations that apply to a specific skill or specialty."""
    for _, scope in _experience_mentions(requirement_text or job.description):
        if job.category == "HEALTHCARE" and _CLINICAL_SCOPE.search(scope):
            return "clinical"
        if scope and (_ROLE_SCOPE.search(scope) or extract_skills(scope)):
            return "specific"
    return None


def _requirements(description: str) -> str:
    required = (
        r"(?:minimum |basic |required )?(?:requirements|qualifications)"
        r"|(?:job |position )requirements|required skills(?: and experience)?"
        r"|what (?:experience you need|you(?:['’]ll| will) (?:bring|need))"
    )
    other = (
        r"preferred (?:skills|qualifications|experience)|nice to have|bonus (?:skills|qualifications)"
        r"|benefits|company overview|about (?:us|the company)|equal opportunity"
        r"|what could set you apart|responsibilities|what you(?:['’]ll| will) do"
    )
    headings = list(re.finditer(
        rf"(?im)(?:^|(?<=[.\n]))[ \t]*(?:(?P<required>{required})|(?:{other}))"
        r"[ \t]*(?::[ \t]*|(?=\r?$))", description,
    ))
    return "\n".join(
        description[heading.end():headings[i + 1].start() if i + 1 < len(headings) else len(description)].strip()
        for i, heading in enumerate(headings) if heading.group("required")
    )


def _requirement_evidence(text: str, have: dict[str, str]) -> tuple[list[str], list[str], int | None]:
    """Extract conservative evidence; alternatives are not all mandatory."""
    required, missing = [], []
    for segment in re.split(r"[\n.;]", text):
        skills = extract_skills(segment)
        required.extend(skills)
        # When one option in an explicit alternatives clause is present, do not
        # present other options in that clause as missing requirements.
        alternative_satisfied = bool(len(segment) <= 250 and re.search(r"\b(?:or|either)\b", segment, re.I) and
                                     any(s.casefold() in have for s in skills))
        if not alternative_satisfied:
            missing.extend(s for s in skills if s.casefold() not in have)
    # Only report explicit experience requirements, not duration in degrees,
    # and leave unclear / optional requirements unknown.
    return sorted(set(required)), sorted(set(missing)), _experience_years(text)


def adjust(profile: Profile, job: Job, judgement: Judgement) -> Judgement:
    """Break ties with observable evidence and flag skills absent from the resume."""
    have = {skill.casefold(): skill for skill in profile.skills}
    mentioned = {s.casefold(): s for s in job.skills}
    matched = sorted((have[key] for key in have.keys() & mentioned.keys()), key=str.casefold)
    requirement_text = _requirements(job.description)
    required, missing, required_years = _requirement_evidence(requirement_text, have)
    if required_years is None and not requirement_text:
        required_years = _experience_years(job.description)
    required_matches = [skill for skill in required if skill.casefold() in have]

    # Prefer demonstrated coverage of skills in the requirements section; absent
    # evidence is not proof the candidate lacks a skill. Cap its influence.
    score = judgement.score
    if required:
        coverage = len(required_matches) / len(required)
        score += 1.5 * (coverage - 0.5)
    if matched:
        score += min(len(matched), 5) * 0.12
    if profile.category != job.category and not matched:
        score -= 2.5
    if required_years is not None and profile.years_experience is not None:
        score -= min(3.0, max(0, required_years - profile.years_experience) * 0.45)
    elif required_years is not None and profile.early_career and required_years >= 3:
        score -= min(2.5, required_years * 0.5)
    senior_title = bool(re.search(r"\b(?:senior|sr\.?|principal|architect|director|head of|chief)\b", job.title, re.I))
    junior_description = bool(re.search(r"\b(?:seeking|looking for|hiring)\b[^.\n]{0,80}\b(?:junior|entry.level)\b", job.description[:1500], re.I))
    senior_gap = profile.early_career and senior_title and not junior_description
    if senior_gap:
        score = min(score - 2.5, 5.9)
    # Treat explicit senior leadership titles cautiously when the profile gives
    # limited experience and does not identify comparable leadership work.
    leadership = re.search(r"\b(?:manager|director|principal|head|chief|lead)\b", roles := " ".join(profile.roles), re.I)
    senior_leadership = re.search(r"\b(?:senior manager|directore?|principal|head of|chief)\b", job.title, re.I)
    if profile.years_experience is not None and profile.years_experience <= 5 and senior_leadership and not leadership:
        score = min(score, 5.9)
    # A software resume is not evidence of network engineering experience.
    roles = roles.casefold()
    network_evidence = roles + " " + " ".join(profile.skills).casefold()
    network_gap = bool(re.search(r"\bnetwork\s+(?:engineer|administrator|architect)\b", job.title, re.I) and not re.search(r"\b(?:network\w*|cisco|routing|switching)\b", network_evidence))
    if network_gap:
        score = min(score, 4.9)
    score = round(max(0.0, min(10.0, score)), 1)
    parts = []
    if matched:
        parts.append("Skills mentioned in both resume and posting: " + ", ".join(matched) + ".")
    else:
        parts.append("No shared listed skills were identified.")
    if missing:
        parts.append("Skills mentioned in the requirements section but not detected in the extracted profile: " + ", ".join(missing) + ".")
    else:
        parts.append("Review the full requirements; missing qualifications may not be detected.")
    if profile.category != job.category and not matched:
        parts.append("The job category differs from the profile and no shared skills were detected; review the occupation.")
    if required_years is not None and profile.years_experience is not None and required_years > profile.years_experience:
        parts.append(f"The posting mentions {required_years} years of experience; the profile states {profile.years_experience:g}. Check the original wording.")
    elif required_years is not None and profile.years_experience is None:
        parts.append(f"The posting mentions {required_years} years of experience; total experience was not established from the resume. Check the original wording.")
    if senior_gap:
        parts.append("The resume indicates an early-career job search, while the title suggests a senior role; check the posting's actual level.")
    if senior_leadership and profile.years_experience is not None and profile.years_experience <= 5 and not leadership:
        parts.append("The job title suggests senior leadership, which was not identified in the profile.")
    if network_gap:
        parts.append("No network engineering role or networking skills were detected in the profile.")
    return Judgement(job.job_id, score, " ".join(parts))


def assessable_preferences(job, preferences):
    """Pass the model only preference fields that have comparable job data."""
    from hiremind.schemas import Preferences
    preferences = preferences or Preferences()
    location = preferences.location if job.location and not re.search(r"\bremote\b", job.location, re.I) else None
    job_hint = " ".join(filter(None, (job.title, job.location)))
    mode_found = next((m for m in ("remote", "hybrid", "onsite") if re.search(r"\b" + m + r"\b", job_hint, re.I)), None)
    return Preferences(location=location, work_mode=preferences.work_mode if mode_found else None)


def early_seniority_conflict(profile: Profile, job: Job) -> bool:
    """Flag a clear senior title for an explicitly early-career job search."""
    if not profile.early_career:
        return False
    senior = re.search(r"\b(?:senior|sr\.?|principal|director|head of|chief)\b|\bstaff\s+(?:software\s+)?(?:engineer|developer|architect)\b", job.title, re.I) or re.search(
        r"\b(?:seeking|looking for|hiring)\b[^.\n]{0,80}\b(?:senior|principal|director)\b", job.description[:1500], re.I)
    junior = re.search(r"\b(?:junior|entry.level)\b", job.title, re.I) or re.search(
        r"\b(?:seeking|looking for|hiring)\b[^.\n]{0,80}\b(?:junior|entry.level)\b", job.description[:1500], re.I)
    return bool(senior and not junior)


def missing_core_tools(profile: Profile, job: Job) -> list[str]:
    """Flag a small set of explicitly named core platforms, not optional mentions."""
    tools = {"Wix Studio": r"\bwix(?:\s+studio)?\b", "React": r"\breact(?:\.?js)?\b", "Angular": r"\bangular(?:js)?\b", "Swift": r"\bswift\b", "Java": r"\bjava\b",
             "Splunk": r"\bsplunk\b", "AI/ML": r"\b(?:ai\s*/\s*ml|machine learning)\b",
             "Shopify": r"\bshopify\b", ".NET": r"\.net\b", "DevOps": r"\bdevops\b", "Node.js": r"\bnode(?:\.?js|\s+js)\b",
             "Salesforce": r"\bsalesforce\b", "Ext JS": r"\bext\s?js\b"}
    known = " ".join(profile.skills).casefold()
    missing = []
    for label, pattern in tools.items():
        if re.search(pattern, known, re.I):
            continue
        explicit_title = re.search(r"\bjob\s+title\s*:\s*([^\n.]{1,80})", job.description[:400], re.I)
        titles = [job.title, explicit_title.group(1) if explicit_title else ""]
        title_hit = any(re.search(pattern, title, re.I) and not re.search(r"\b(?:or|either)\b", title, re.I)
                        for title in titles)
        hard_requirement = False
        for hit in re.finditer(pattern, job.description, re.I):
            before = job.description[max(0, hit.start() - 55):hit.start()]
            after = job.description[hit.end():hit.end() + 55]
            if re.search(r"^\s*(?:or\b|[^.\n]{0,25}\b(?:preferred|optional|is a plus)\b)", after, re.I):
                continue
            if (re.search(r"\b(?:must|essential|speciali[sz]ed in|proven experience|requires?|well versed in|proficiency in)\b[^.\n]{0,40}$", before, re.I)
                    or re.search(r"^[^.\n]{0,25}\b(?:required(?:\b|(?-i:(?=[A-Z])))|is a must\b)", after, re.I)):
                hard_requirement = True
                break
        if title_hit or hard_requirement:
            missing.append(label)
    return missing


_CREDENTIALS = {
    "CPA": re.compile(r"\b(?:CPA|certified public accountant)\b", re.I),
    "RN": re.compile(r"\b(?:RN|registered nurs(?:e|ing))\b", re.I),
    "LPN/LVN": re.compile(r"\b(?:LPN|LVN|licensed (?:practical|vocational) nurs(?:e|ing))\b", re.I),
    "CNA": re.compile(r"\b(?:CNA|certified nursing assistant)\b", re.I),
}


def missing_title_credentials(profile: Profile, job: Job) -> list[str]:
    """Flag explicit credential titles when the resume gives no corresponding evidence."""
    named = {name for name, pattern in _CREDENTIALS.items() if pattern.search(job.title)}
    if not named:
        return []
    evidence = " ".join([*profile.roles, *profile.skills, *profile.education])
    supported = {name for name in named if _CREDENTIALS[name].search(evidence)}
    if len(named) > 1 and supported and re.search(r"\bor\b|/", job.title, re.I):
        return []
    return sorted(named - supported)


def score_weighted(profile, job, requirements_score, preferences=None, mode="standard", focused_notes=(),
                   experience_override=None, preference_override=None):
    """Provisional 60/30/10 score; unknown components are never zero-filled."""
    from hiremind.schemas import Preferences, ScoreBreakdown

    preferences = preferences or Preferences()
    have = {s.casefold(): s for s in profile.skills}
    shared = sorted((s for s in job.skills if s.casefold() in have), key=str.casefold)
    text = _requirements(job.description)
    required, missing, years = _requirement_evidence(text, have)
    if years is None and not text:
        years = _experience_years(job.description)
    notes = []
    req = float(requirements_score)
    if required:
        # The extractor suppresses other options in a satisfied alternatives
        # clause; do not penalize them again as individually required skills.
        coverage = (len(required) - len(missing)) / len(required)
        req += 1.5 * (coverage - .5)
    if profile.category not in {config.UNKNOWN_CATEGORY, job.category} and not shared:
        req -= 2.5
    if profile.category == config.UNKNOWN_CATEGORY and not profile.roles and not shared:
        req = min(req, 5.9)
    roles_and_skills = " ".join([*profile.roles, *profile.skills])
    network_gap = bool(re.search(r"\bnetwork\s+(?:engineer|administrator|architect)\b", job.title, re.I) and
                       not re.search(r"\b(?:network\w*|cisco|routing|switching)\b", roles_and_skills, re.I))
    if network_gap:
        req = min(req, 4.9)
    core_missing = missing_core_tools(profile, job)
    if core_missing:
        req = min(req, 5.9)
    credentials = missing_title_credentials(profile, job)
    if credentials:
        req = min(req, 5.9)
    req = round(max(0, min(10, req)), 1)
    notes.append("Shared skills: " + ", ".join(shared) + "." if shared else "No shared listed skills detected.")
    if profile.category == config.UNKNOWN_CATEGORY:
        notes.append("Résumé category is unknown. Check the evidence in the resume before relying on this score.")
    if missing:
        notes.append("Skills mentioned in requirements but not detected in the profile: " + ", ".join(missing) + ".")
    else:
        notes.append("Review the posting; the requirements extractor may miss qualifications.")
    if network_gap:
        notes.append("No network engineering role or networking skills were detected in the profile; verify the occupation.")
    if core_missing:
        notes.append("Core tools named in the job title or an explicit requirement were not detected in the profile: " + ", ".join(core_missing) + ".")
    if credentials:
        notes.append("Credentials named in the job title with no corresponding resume evidence: " + ", ".join(credentials) + ".")
    experience = None
    scope = experience_scope(job, text)
    if scope == "clinical":
        notes.append("The posting asks for experience in a clinical specialty. Total résumé years cannot establish time in that specialty.")
    elif scope == "specific":
        notes.append("The posting asks for experience with a specific role or skill. Total résumé years do not establish the duration of that experience.")
    elif years is not None and profile.years_experience is not None:
        gap = max(0, years - profile.years_experience)
        experience = round(max(0, 10 - min(10, gap * 2)), 1)
        notes.append(f"Posting mentions {years} years; profile states {profile.years_experience:g}. Verify context.")
        if experience_override is not None:
            experience = round(min(float(experience_override), experience + 1, 10), 1)
            notes.append("In-depth experience assessment used, constrained by the stated experience gap.")
    else:
        notes.append("Comparable experience durations were not established from the resume and posting.")
        if profile.early_career and years is not None and years >= 3:
            experience = 3.5
            notes.append("The posting's experience requirement is high for the stated early-career search; actual total years remain unknown.")
    if early_seniority_conflict(profile, job):
        experience = min(experience, 3.5) if experience is not None else 3.5
        notes.append("Seniority estimate uses the early-career intent and job title; experience years may be unknown and the title may be wrong.")
    leadership_title = re.search(r"\b(?:senior manager|directore?|principal|head of|chief)\b", job.title, re.I)
    leadership_role = re.search(r"\b(?:manager|director|principal|head|chief|lead)\b", " ".join(profile.roles), re.I)
    if leadership_title and profile.years_experience is not None and profile.years_experience <= 5 and not leadership_role:
        experience = min(experience, 4.0) if experience is not None else 4.0
        notes.append("Seniority estimate is based on the leadership title and profile; verify the actual responsibilities.")
    preference_parts = []
    if preferences.location:
        if job.location and not re.search(r"\bremote\b", job.location, re.I):
            value = 10.0 if preferences.location.casefold().strip() in job.location.casefold() else 0.0
            preference_parts.append(value)
            notes.append(f"Location: {job.location} (requested: {preferences.location}).")
        else:
            notes.append("Location preference could not be assessed; check remote eligibility and location.")
    if preferences.work_mode:
        job_hint = " ".join(filter(None, (job.title, job.location)))
        mode_found = next((m for m in ("remote", "hybrid", "onsite") if re.search(r"\b" + m + r"\b", job_hint, re.I)), None)
        if mode_found:
            preference_parts.append(10.0 if mode_found == preferences.work_mode.casefold() else 0.0)
            notes.append(f"Work arrangement in title/location: {mode_found}; verify the posting.")
        else:
            notes.append("Work arrangement is not reliably available in the job title or location.")
    if preferences.min_salary is not None:
        notes.append("Salary preference is unscored: this job dataset has no structured salary information.")
    if preferences.benefits:
        notes.append("Benefits preference is unscored: this job dataset has no reliable structured benefits information.")
    pref = round(sum(preference_parts) / len(preference_parts), 1) if preference_parts else None
    if pref is not None and preference_override is not None:
        notes.append(f"In-depth model preference review: {float(preference_override):g}/10; the preference score uses the explicit comparison. Verify location and work arrangement.")
    requested = sum((bool(preferences.location), bool(preferences.work_mode), preferences.min_salary is not None, bool(preferences.benefits)))
    preference_coverage = len(preference_parts) / requested if requested else 0.0
    components = [(req, .6), (experience, .3), (pref, .1 * preference_coverage)]
    available = sum(w for value, w in components if value is not None)
    score = round(sum(value * w for value, w in components if value is not None) / available, 1)
    q_weight = .6 + (.3 if experience is not None else 0)
    qualification = round((req * .6 + (experience or 0) * .3) / q_weight, 1)
    notes.extend(focused_notes)
    breakdown = ScoreBreakdown(req, experience, pref, qualification, round(available, 2), notes, mode, round(preference_coverage, 2))
    return Judgement(job.job_id, score, " ".join(notes), breakdown)
