"""Create a grounded, editable business-style cover letter.

The draft is assembled from the job title and verified profile fields. It does
not invent contact details, employers, achievements, or a hiring contact.
"""
from dataclasses import asdict
import json
from pathlib import Path
import re

from hiremind import slm
from hiremind.schemas import Job, Profile
from hiremind.skills import extract_skills


def build_prompt(profile: Profile, job: Job) -> str:
    """Format evidence for Writer training and offline evaluation."""
    return ("Write an editable business cover letter using only the profile evidence and the supplied job title. "
            "Leave contact and employer details as placeholders.\n"
            + json.dumps({"profile": asdict(profile), "job_title": job.title, "job_skills": job.skills}, ensure_ascii=False))


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


_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "cover-letter" / "SKILL.md"
_ACTION = re.compile(
    r"^(?:build|develop|design|maintain|manage|support|analy[sz]e|prepare|deliver|lead|create|implement|"
    r"write|automate|improve|review|coordinate|monitor|provide|document|collaborate|test|troubleshoot|own|"
    r"work with|work on|assist|conduct|partner|produce|use|gather|perform|deploy)\b", re.I,
)
_PAST_ACTIONS = {
    "built", "created", "developed", "designed", "analyzed", "analysed", "implemented", "updated",
    "worked", "wrote", "prepared", "supported", "maintained", "assessed", "documented", "provided",
    "coordinated", "managed", "used", "automated", "led", "tested", "delivered", "improved",
    "reconciled", "assisted", "reviewed", "produced", "deployed",
}


def _unique_lines(values):
    lines = (" ".join(value.split()).lstrip("-• ").rstrip(". ") for value in values)
    return list(dict.fromkeys(line for line in lines if line))


def _posting_focus(profile: Profile, job: Job) -> list[str]:
    """Surface concise duties first, then specific skills listed in the posting."""
    description = re.sub(r"(?m)^[^\S\n]*(?:\d+[.)]|[-*•–—])[^\S\n]+", "", job.description[:12000])
    parts = re.split(r"[\n;•]+|(?<=[.!?])\s+(?=[A-Z])", description)
    duties = [line for line in _unique_lines(parts) if 22 <= len(line) <= 165 and _ACTION.match(line)]
    job_skills = _unique_lines([*job.skills, *extract_skills(job.description[:12000])])
    profile_skills = {skill.casefold() for skill in profile.skills}
    duties.sort(key=lambda line: -len(profile_skills & {skill.casefold() for skill in extract_skills(line)}))
    covered = {skill.casefold() for duty in duties[:6] for skill in extract_skills(duty)}
    job_skills = [skill for skill in job_skills if skill.casefold() not in covered]
    job_skills.sort(key=lambda skill: skill.casefold() not in profile_skills)
    return _unique_lines([*duties[:6], *job_skills[:5]])


def _select_connections(answer: str, evidence_count: int, focus_count: int) -> list[tuple[int, int, str]]:
    plan = json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
    rows = plan["connections"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
        raise ValueError("Expected one or two evidence connections")
    selected = []
    for row in rows:
        evidence, focus, style = row["evidence"], row["focus"], row["style"]
        if (type(evidence) is not int or type(focus) is not int
                or not 1 <= evidence <= evidence_count or not 1 <= focus <= focus_count
                or style not in ("direct", "role_first")
                or any(evidence - 1 == e or focus - 1 == f for e, f, _ in selected)):
            raise ValueError("Invalid or repeated letter connection")
        selected.append((evidence - 1, focus - 1, style))
    return selected


def _alignment(example: str, duty: str) -> int:
    def terms(text):
        return {word.removesuffix("s") for word in re.findall(r"[a-z][a-z0-9]+", text.casefold())
                if len(word) > 2 and word not in {"the", "and", "with", "for", "from", "using", "that", "your",
                                                  "work", "team", "weekly", "monthly", "daily", "each"}}
    skills = set(extract_skills(example)) & set(extract_skills(duty))
    common = terms(example) & terms(duty)
    return 3 * len(skills) + (len(common) if len(common) >= 2 else 0)


def _complete_connections(pairs: dict[tuple[int, int], int], selected) -> list[tuple[int, int, str]]:
    selected = [(e, f, style) for e, f, style in selected if (e, f) in pairs]
    for e, f in sorted(pairs, key=pairs.get, reverse=True):
        if len(selected) == 2:
            break
        if all(e != old_e and f != old_f for old_e, old_f, _ in selected):
            selected.append((e, f, "direct"))
    return selected


def _personalized_letter(profile: Profile, job: Job, evidence: list[str], focus: list[str], plan) -> str:
    name = profile.name.strip() if profile.name and profile.name.strip() else "[Your name]"
    opening = f"I am applying for the {job.title.strip() or '[Job title]'} position."

    def experience_sentence(fact):
        if fact.split(maxsplit=1)[0].casefold() in _PAST_ACTIONS:
            return f"I {fact[0].lower() + fact[1:]}."
        return f"My résumé describes this experience: “{fact}.”"

    paragraphs = []
    used_evidence = []
    for e, f, style in plan:
        fact, duty = evidence[e], focus[f]
        used_evidence.append(fact)
        experience = experience_sentence(fact)
        if _ACTION.match(duty) and len(duty) >= 22:
            action = duty[0].lower() + duty[1:]
            if style == "role_first":
                paragraphs.append(f"Your posting asks for someone to {action}. {experience} I would bring that approach to this responsibility.")
            else:
                paragraphs.append(f"{experience} I would draw on that work to help {action}.")
        elif style == "role_first":
            paragraphs.append(f"Your posting highlights {duty}. {experience} I would bring that approach to this work.")
        else:
            paragraphs.append(f"{experience} I would draw on that work to contribute to the role’s focus on {duty}.")
    if not plan:
        context = []
        if evidence:
            used_evidence.append(evidence[0])
            context.append(experience_sentence(evidence[0]))
        elif roles := _unique_lines(profile.roles):
            context.append(f"My résumé lists my role as {roles[0]}.")
        if education := _unique_lines(profile.education):
            context.append(f"My education includes {education[0]}.")
        if context:
            paragraphs.append(" ".join(context))
        if focus and _ACTION.match(focus[0]):
            duty = focus[0]
            paragraphs.append(f"I am interested in the opportunity to {duty[0].lower() + duty[1:]}.")
    used_skills = {skill.casefold() for skill in extract_skills(" ".join(used_evidence))}
    posting_skills = {skill.casefold() for skill in [*job.skills, *extract_skills(job.title + "\n" + job.description[:12000])]}
    additional = [skill for skill in profile.skills if skill.casefold() in posting_skills - used_skills]
    if additional:
        paragraphs.append("My résumé also lists " + ", ".join(additional[:4]) + ".")
    closing = "I would welcome the chance to discuss how I could contribute to the role. Thank you for your consideration."
    return "\n\n".join([name, "[Your email] | [Your phone]", "[Date]", "[Employer name]",
                       "Dear Hiring Manager,", opening, *paragraphs, closing, "Sincerely,", name])


def write_personalized(profile: Profile, job: Job, progress=None) -> str:
    """Ask the local Writer for a plan and render only verified source values."""
    def report(phase):
        if progress:
            progress({"phase": phase, "attempt": 1, "total": 1})

    report("drafting")
    evidence = _unique_lines([*profile.experience_evidence, *profile.projects])
    focus = _posting_focus(profile, job)
    candidates = [(example, {f: score for f, duty in enumerate(focus) if (score := _alignment(example, duty))})
                  for example in evidence]
    candidates.sort(key=lambda candidate: -max(candidate[1].values(), default=0))
    candidates = candidates[:12]
    evidence = [example for example, _ in candidates]
    pairs = {(e, f): score for e, (_, scores) in enumerate(candidates) for f, score in scores.items()}
    answer = ""
    if pairs:
        source = {"job_title": job.title, "evidence": dict(enumerate(evidence, 1)),
                  "posting_focus": dict(enumerate(focus, 1)),
                  "supported_pairs": [[e + 1, f + 1] for e, f in pairs]}
        # Read the project instructions at drafting time so edits take effect
        # without having to reload the model.
        try:
            prompt = _SKILL_PATH.read_text(encoding="utf-8") + "\nInput:\n" + json.dumps(source, ensure_ascii=False)
            answer = slm.generate("writer", prompt, max_new_tokens=160)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError):
            pass
    report("checking")
    try:
        selected = _select_connections(answer, len(evidence), len(focus)) if answer else []
    except (ValueError, KeyError, TypeError):
        selected = []
    plan = _complete_connections(pairs, selected)
    letter = _personalized_letter(profile, job, evidence, focus, plan)
    report("done")
    return letter
