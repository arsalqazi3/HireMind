"""Critic: scores each job out of 10 on category, skills and level."""
import json
import math
from dataclasses import asdict

from hiremind import config, slm
from hiremind.schemas import Job, Judgement, Profile


def judge(profile: Profile, jobs: list[Job]) -> list[Judgement]:
    judgements = []
    for job in jobs:
        prompt = ('Assess how well this profile matches the job. Return a JSON object with a numeric "score" from 0 to 10 '
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


def feedback(profile: Profile, judgements: list[Judgement]) -> str | None:
    """Return advice for the Searcher (for example "too senior, look for junior
    roles") if the best score is below config.CRITIC_CUTOFF, else None."""
    if judgements and max(j.score for j in judgements) >= config.CRITIC_CUTOFF:
        return None
    return "No strong match found; try a different query using the candidate's skills and experience."
