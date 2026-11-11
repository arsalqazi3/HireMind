"""Runs the whole agent: parse, search with Critic retries, rank, write.

The flags turn parts off for the ablation study (evaluation/ablation.py).
"""
from dataclasses import dataclass, field

from hiremind.schemas import Judgement, Profile, Step
from hiremind import config, tools
from hiremind.memory import Memory
from hiremind.roles import critic, parser, searcher, writer


@dataclass
class Result:
    profile: Profile
    top_jobs: list[Judgement]
    steps: list[Step] = field(default_factory=list)  # shown in the app
    retries: int = 0


def run(resume_text: str, use_critic: bool = True, use_episodic: bool = True) -> Result:
    """Parse the resume, then up to config.MAX_SEARCH_ROUNDS rounds of
    Searcher then Critic, and return the top config.TOP_K jobs with reasons."""
    profile = parser.parse(resume_text)
    memory = Memory(semantic=profile, use_episodic=use_episodic)
    ranked: dict[str, Judgement] = {}
    retries = 0
    for round_number in range(config.MAX_SEARCH_ROUNDS):
        jobs = searcher.search(memory)
        if use_critic:
            judgements = critic.judge(profile, jobs)
            ranked.update({j.job_id: j for j in judgements if j.score >= config.CRITIC_CUTOFF})
            advice = critic.feedback(profile, judgements)
        else:
            ranked.update({job.job_id: Judgement(job.job_id, 0, "Retrieved by search; not scored.") for job in jobs})
            advice = None
        if advice is None or round_number + 1 == config.MAX_SEARCH_ROUNDS:
            break
        if memory.use_episodic:
            memory.episodic.critic_feedback.append(advice)
            memory.episodic.rejected.update({j.job_id: j.reason for j in judgements if j.score < config.CRITIC_CUTOFF})
        retries += 1
    return Result(profile, sorted(ranked.values(), key=lambda j: j.score, reverse=True)[:config.TOP_K], memory.short_term, retries)


def cover_letter(profile: Profile, job_id: str) -> str:
    return writer.write_checked(profile, tools.get_job_details(job_id))
