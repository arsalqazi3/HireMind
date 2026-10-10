"""Runs the whole agent: parse, search with Critic retries, rank, write.

The flags turn parts off for the ablation study (evaluation/ablation.py).
"""
from dataclasses import dataclass, field

from hiremind.schemas import Judgement, Profile, Step


@dataclass
class Result:
    profile: Profile
    top_jobs: list[Judgement]
    steps: list[Step] = field(default_factory=list)  # shown in the app
    retries: int = 0


def run(resume_text: str, use_critic: bool = True, use_episodic: bool = True) -> Result:
    """Parse the resume, then up to config.MAX_SEARCH_ROUNDS rounds of
    Searcher then Critic, and return the top config.TOP_K jobs with reasons."""
    raise NotImplementedError


def cover_letter(profile: Profile, job_id: str) -> str:
    raise NotImplementedError
