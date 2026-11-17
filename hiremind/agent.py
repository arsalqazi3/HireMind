"""Runs the whole agent: parse, search with Critic retries, rank, write.

The flags turn parts off for the ablation study (evaluation/ablation.py).
"""
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable

from hiremind.schemas import Judgement, Preferences, Profile, Step
from hiremind import config, matching, tools
from hiremind.memory import Memory
from hiremind.roles import critic, parser, searcher, writer


@dataclass
class Result:
    profile: Profile
    top_jobs: list[Judgement]
    steps: list[Step] = field(default_factory=list)  # shown in the app
    retries: int = 0


def run(resume_text: str, use_critic: bool = True, use_episodic: bool = True,
        progress: Callable[[dict], None] | None = None, preferences: Preferences | None = None,
        in_depth: bool = False) -> Result:
    """Parse the resume, then up to config.MAX_SEARCH_ROUNDS rounds of
    Searcher then Critic, and return the top config.TOP_K jobs with reasons."""
    def report(**event):
        if progress:
            progress(event)

    report(phase="parsing")
    profile = parser.parse(resume_text)
    memory = Memory(semantic=profile, use_episodic=use_episodic)
    ranked: dict[str, Judgement] = {}
    scored_ids: set[str] = set()
    retries = 0
    for round_number in range(config.MAX_SEARCH_ROUNDS):
        report(phase="searching", round=round_number + 1)
        jobs = searcher.search(memory)
        if jobs and use_episodic:
            jobs = [job for job in jobs if job.job_id not in scored_ids]
            if not jobs:
                break
        if not jobs:
            role = profile.roles[0] if profile.roles else profile.category.replace("-", " ")
            options = [" ".join(" ".join([role, *skills]).split()[:8]) for skills in
                       (profile.skills[:3], profile.skills[-3:], profile.skills[:1], [])]
            tried = {q.casefold() for q in memory.episodic.queries}
            query = next((q for q in options if q and q.casefold() not in tried), "")
            if query and query.casefold() not in tried:
                found = tools.search_jobs(query)
                jobs = [job for job in found if not use_episodic or job.job_id not in memory.episodic.seen_job_ids]
                if use_episodic:
                    memory.episodic.queries.append(query)
                    memory.episodic.seen_job_ids.update(job.job_id for job in jobs)
                memory.short_term.append(Step(state={}, action="search_jobs", args={"query": query},
                                              result=[{"job_id": job.job_id, "title": job.title} for job in jobs],
                                              feedback="Used an alternative profile-based query after an empty search."))
        if not jobs:
            break
        if use_episodic:
            scored_ids.update(job.job_id for job in jobs)
        if use_critic:
            judgements = []
            started = monotonic()
            report(phase="scoring", round=round_number + 1, completed=0, total=len(jobs), remaining_seconds=None)
            for completed, job in enumerate(jobs, 1):
                focused_notes = []
                if in_depth:
                    requirement_score = critic.focused(profile, job, "requirements", preferences)
                    if requirement_score is None:
                        judged = critic.judge(profile, [job])
                        requirement_score = judged[0].score if judged else None
                        focused_notes.append("Focused requirements output unavailable; standard assessment used when possible.")
                else:
                    judged = critic.judge(profile, [job])
                    requirement_score = judged[0].score if judged else None
                if requirement_score is not None:
                    mode = "in-depth" if in_depth else "standard"
                    initial = matching.score_weighted(profile, job, requirement_score, preferences, mode)
                    exp_score = pref_score = None
                    if in_depth:
                        if initial.breakdown.experience is not None and profile.years_experience is not None and matching._experience_years(matching._requirements(job.description) or job.description) is not None:
                            exp_score = critic.focused(profile, job, "experience", preferences)
                            if exp_score is None:
                                focused_notes.append("Focused experience output unavailable; the stated-years comparison was used.")
                        else:
                            focused_notes.append("Focused experience check skipped: comparable stated years are unavailable.")
                        if initial.breakdown.preferences is not None:
                            pref_score = critic.focused(profile, job, "preferences", matching.assessable_preferences(job, preferences))
                            if pref_score is None:
                                focused_notes.append("Focused preference output unavailable; available preference facts were used.")
                        else:
                            focused_notes.append("Focused preference check skipped: comparable preferences are unavailable.")
                    judgements.append(matching.score_weighted(profile, job, requirement_score, preferences, mode,
                                      focused_notes, exp_score, pref_score))
                elapsed = monotonic() - started
                report(phase="scoring", round=round_number + 1, completed=completed, total=len(jobs),
                       remaining_seconds=elapsed / completed * (len(jobs) - completed) if completed >= 2 else None)
            jobs_by_id = {job.job_id: job for job in jobs}
            eligible = [j for j in judgements if j.breakdown.requirements >= config.CRITIC_CUTOFF
                        and j.breakdown.qualification >= config.CRITIC_CUTOFF
                        and not matching.early_seniority_conflict(profile, jobs_by_id[j.job_id])]
            ranked.update({j.job_id: j for j in eligible})
            qualification_judgements = [Judgement(j.job_id, j.breakdown.qualification, j.reason) for j in eligible]
            advice = critic.feedback(profile, qualification_judgements)
            if advice is None and len(ranked) < config.TOP_K:
                advice = "Fewer than five suitable jobs found; search for additional roles and skills."
        else:
            ranked.update({job.job_id: Judgement(job.job_id, 0, "Retrieved by search; not scored.") for job in jobs})
            advice = None
        if advice is None or round_number + 1 == config.MAX_SEARCH_ROUNDS:
            break
        if memory.use_episodic:
            memory.episodic.critic_feedback.append(advice)
            memory.episodic.rejected.update({j.job_id: j.reason for j in judgements if j.score < config.CRITIC_CUTOFF})
        retries += 1
    report(phase="done")
    return Result(profile, sorted(ranked.values(), key=lambda j: j.score, reverse=True)[:config.TOP_K], memory.short_term, retries)


def cover_letter(profile: Profile, job_id: str, progress=None) -> str:
    return writer.write_checked(profile, tools.get_job_details(job_id), progress=progress)
