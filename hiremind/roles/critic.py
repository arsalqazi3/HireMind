"""Critic: scores each job out of 10 on category, skills and level."""
import json
import math
from dataclasses import asdict

from hiremind import config, slm
from hiremind.schemas import Job, Judgement, Profile


def judge(profile: Profile, jobs: list[Job]) -> list[Judgement]:
    judgements = []
    for job in jobs:
        prompt = ('Assess only the match between the profile and job duties, required skills and qualifications. Do not score salary, location, benefits or years of experience. Return a JSON object with a numeric "score" from 0 to 10 '
                  'and a "reason" explaining the actual matching skills and any gaps. Do not invent qualifications.\nProfile: ' + json.dumps(asdict(profile), ensure_ascii=False)
                  + '\nJob: ' + json.dumps(asdict(job), ensure_ascii=False))
        answer = slm.generate("critic", prompt, max_new_tokens=200).strip()
        try:
            result = json.loads(answer.removeprefix("```json").removesuffix("```").strip())
            score, reason = result["score"], result["reason"]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10 or not isinstance(reason, str) or not reason.strip() or reason.strip().casefold() == "brief evidence and gaps":
                continue
            judgements.append(Judgement(job.job_id, float(score), reason.strip()))
        except (ValueError, TypeError, KeyError):
            continue
    return judgements


def focused(profile: Profile, job: Job, dimension: str, preferences=None) -> float | None:
    """Separate optional model check; invalid output is explicitly unknown."""
    instructions = {
        "requirements": "Assess only job duties and required skills; do not score years, salary or preferences.",
        "experience": "Assess only the stated years of experience against the posting's stated experience requirement. Do not infer missing years.",
        "preferences": "Assess only the stated location and work arrangement preferences against the posting. Never infer missing salary or location.",
    }
    prompt = (instructions[dimension] + ' Return only JSON with a numeric "score" from 0 to 10 and "reason". '
              'Use null for the score if the supplied information is insufficient.\nProfile: '
              + json.dumps(asdict(profile), ensure_ascii=False) + '\nJob: ' + json.dumps(asdict(job), ensure_ascii=False)
              + '\nPreferences: ' + json.dumps(asdict(preferences) if preferences else {}, ensure_ascii=False))
    try:
        answer = slm.generate("critic", prompt, max_new_tokens=200).strip()
        data = json.loads(answer.removeprefix("```json").removesuffix("```").strip())
        score = data["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
            return None
        return float(score)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def feedback(profile: Profile, judgements: list[Judgement]) -> str | None:
    """Return advice for the Searcher (for example "too senior, look for junior
    roles") if the best score is below config.CRITIC_CUTOFF, else None."""
    if judgements and max(j.score for j in judgements) >= config.CRITIC_CUTOFF:
        return None
    return "No strong match found; try a different query using the candidate's skills and experience."
