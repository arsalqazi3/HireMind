"""Critic: scores each job out of 10 on category, skills and level."""
from hiremind.schemas import Job, Judgement, Profile


def judge(profile: Profile, jobs: list[Job]) -> list[Judgement]:
    raise NotImplementedError


def feedback(profile: Profile, judgements: list[Judgement]) -> str | None:
    """Return advice for the Searcher (for example "too senior, look for junior
    roles") if the best score is below config.CRITIC_CUTOFF, else None."""
    raise NotImplementedError
