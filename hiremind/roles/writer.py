"""Create a grounded, editable business-style cover letter.

The draft is assembled from the job title and verified profile fields. It does
not invent contact details, employers, achievements, or a hiring contact.
"""
from hiremind.schemas import Job, Profile
from hiremind.skills import extract_skills


def _experience_sentences(profile: Profile, job: Job | None) -> list[str]:
    """Select literal resume bullets, preferring skills present in the posting."""
    job_terms = {s.casefold() for s in (job.skills if job else [])}
    if job:
        job_terms.update(s.casefold() for s in extract_skills(job.description))
    candidates = [item.strip().rstrip(". ") for item in profile.experience_evidence if item.strip()]
    candidates.sort(key=lambda text: -len({s.casefold() for s in extract_skills(text)} & job_terms))
    sentences = []
    for text in candidates[:2]:
        first = text.split(maxsplit=1)[0].casefold()
        if first in {"built", "created", "developed", "designed", "analyzed", "analysed", "implemented", "updated", "worked", "wrote", "prepared", "supported", "maintained", "assessed", "documented", "provided", "coordinated", "managed", "used"}:
            sentences.append("I " + text[0].lower() + text[1:] + ".")
        else:
            sentences.append("My resume describes this experience: “" + text + ".\u201d")
    return sentences


def _safe_lines(profile: Profile, job: Job | None = None) -> list[str]:
    title = job.title.strip() if job and job.title.strip() else "[Job title]"
    name = profile.name.strip() if profile.name and profile.name.strip() else "[Your name]"
    lines = [name, "[Your email] | [Your phone]", "[Date]", "[Employer name]", "Dear Hiring Manager,"]
    opening = f"I am writing to apply for the {title} position."
    if profile.years_experience is not None:
        opening += f" I have {profile.years_experience:g} years of experience."
    lines.append(opening)
    evidence = []
    if profile.roles:
        evidence.append("My resume highlights these roles: " + ", ".join(profile.roles[:3]) + ".")
    job_skills = {s.casefold() for s in job.skills} if job else set()
    relevant = [s for s in profile.skills if s.casefold() in job_skills]
    if relevant:
        evidence.append("My skills include " + ", ".join(relevant[:6]) + ".")
    elif profile.skills:
        evidence.append("My skills include " + ", ".join(profile.skills[:6]) + ".")
    experience = _experience_sentences(profile, job)
    if evidence:
        lines.append(" ".join(evidence))
    if experience:
        lines.append(" ".join(experience))
    evidence = []
    if profile.projects and profile.projects[0].strip():
        project = profile.projects[0].strip().rstrip(". ")
        if project.split(maxsplit=1)[0].casefold() in {"built", "created", "developed", "designed", "analyzed", "implemented"}:
            evidence.append("In one project, I " + project[0].lower() + project[1:] + ".")
        else:
            evidence.append("My resume also describes this project: " + project + ".")
    if evidence:
        lines.append(" ".join(evidence))
    if profile.education:
        lines.append("My education includes " + ", ".join(profile.education[:2]) + ".")
    lines.extend(["I would welcome the opportunity to discuss my qualifications for this role. Thank you for your consideration.",
                  "Sincerely,", name])
    return lines


def unsupported_claims(letter: str, profile: Profile, job: Job | None = None) -> list[str]:
    """Identify lines outside the deterministic draft; this is not a general claim verifier."""
    allowed = set(_safe_lines(profile, job))
    return [line.strip() for line in letter.splitlines() if line.strip() and line.strip() not in allowed]


def write_checked(profile: Profile, job: Job, max_tries: int = 2, progress=None) -> str:
    """Compose a draft from supported fields. The user must review and edit it."""
    if max_tries < 1:
        raise ValueError("max_tries must be positive")
    if progress:
        progress({"phase": "drafting", "attempt": 1, "total": 1})
    letter = "\n\n".join(_safe_lines(profile, job))
    if progress:
        progress({"phase": "checking", "attempt": 1, "total": 1})
    if unsupported_claims(letter, profile, job):
        raise ValueError("Could not verify the generated draft")
    if progress:
        progress({"phase": "done", "attempt": 1, "total": 1})
    return letter
