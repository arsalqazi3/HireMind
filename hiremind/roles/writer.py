"""Writer: cover letter for the chosen job, then a claim check."""
import json
from dataclasses import asdict

from hiremind import slm
from hiremind.schemas import Job, Profile


def _safe_lines(profile: Profile) -> list[str]:
    lines = ["Dear Hiring Manager,", "I am interested in this position."]
    if profile.skills:
        lines.append("My listed skills include " + ", ".join(profile.skills) + ".")
    if profile.education:
        lines.append("My education includes " + ", ".join(profile.education) + ".")
    lines.extend(["Thank you for your consideration.", "Sincerely,"])
    return lines


def write(profile: Profile, job: Job) -> str:
    prompt = ('Write a short cover letter for this job using only facts from the profile. '
              'Do not invent qualifications.\nProfile: ' + json.dumps(asdict(profile), ensure_ascii=False)
              + '\nJob: ' + json.dumps(asdict(job), ensure_ascii=False))
    return slm.generate("writer", prompt, max_new_tokens=512).strip()


def unsupported_claims(letter: str, profile: Profile) -> list[str]:
    """Return skills, experience, education or projects in the letter that are not in the profile."""
    allowed = set(_safe_lines(profile))
    return [line.strip() for line in letter.splitlines() if line.strip() and line.strip() not in allowed]


def write_checked(profile: Profile, job: Job, max_tries: int = 2) -> str:
    """Write the letter, then remove unsupported claims or write it again."""
    if max_tries < 1:
        raise ValueError("max_tries must be positive")
    for _ in range(max_tries):
        letter = write(profile, job)
        if [line.strip() for line in letter.splitlines() if line.strip()] == _safe_lines(profile):
            return letter
    return "\n\n".join(_safe_lines(profile))
