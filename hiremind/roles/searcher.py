"""Searcher: ReAct loop that picks tools until it has candidate jobs.

Reads the profile from semantic memory, and past queries and Critic feedback
from episodic memory, so it does not repeat a failed search.
"""
import json
from dataclasses import asdict

from hiremind import slm, tools
from hiremind.memory import Memory
from hiremind.schemas import Job, Step


def next_action(memory: Memory) -> tuple[str, dict]:
    """Ask the SLM for the next tool name and its arguments, or ("stop", {})."""
    if memory.semantic is None:
        raise ValueError("A profile is required before searching")
    state = {"profile": asdict(memory.semantic)}
    if memory.use_episodic:
        state.update(queries=memory.episodic.queries, feedback=memory.episodic.critic_feedback)
    state["recent_steps"] = [asdict(step) for step in memory.short_term[-4:]]
    prompt = (
        'Choose the next search action. Return only JSON with "tool" and "arguments". '
        'Actions: {"tool":"search_jobs","arguments":{"query":"job and skills"}} '
        '(query at most 8 words), {"tool":"filter_jobs","arguments":{"category":"category or null","level":"level or null"}}, '
        '{"tool":"get_job_details","arguments":{"job_id":"id"}}, '
        '{"tool":"skill_overlap","arguments":{"job_id":"id"}}, or '
        '{"tool":"stop","arguments":{}}. Search first; stop once you have candidates. '
        'Do not repeat previous queries. State:\n' + json.dumps(state, ensure_ascii=False)
    )
    if not memory.short_term and not (memory.use_episodic and memory.episodic.queries):
        prompt = (
            'Find jobs for this profile. Return only JSON in this exact structure: '
            '{"tool":"search_jobs","arguments":{"query":"job and skills"}}. '
            'Replace "job and skills" with a search query of at most 8 words. '
            'Do not repeat words or include an explanation.\nProfile:\n' + json.dumps(state["profile"], ensure_ascii=False)
        )
    answer = slm.generate("searcher", prompt, max_new_tokens=128).strip()
    try:
        action = json.loads(answer.removeprefix("```json").removesuffix("```").strip())
        name, args = action["tool"], action["arguments"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Invalid search action JSON") from exc
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ValueError("Search action arguments must be an object")
    if name == "search_jobs" and set(args) == {"query"} and isinstance(args["query"], str) and 0 < len(args["query"].split()) <= 8:
        query = " ".join(args["query"].split())
        if query.casefold() == "job and skills":
            raise ValueError("Search query is an unfilled example")
        if memory.use_episodic and query.casefold() in {q.casefold() for q in memory.episodic.queries}:
            raise ValueError("Repeated search query")
        return name, {"query": query}
    if name == "filter_jobs" and set(args) <= {"category", "level"} and all(v is None or isinstance(v, str) for v in args.values()):
        return name, args
    if name in {"get_job_details", "skill_overlap"} and set(args) == {"job_id"} and isinstance(args["job_id"], str):
        return name, args
    if name == "stop" and not args:
        return name, args
    raise ValueError("Invalid search action or arguments")


def search(memory: Memory) -> list[Job]:
    """Run one search round and return the candidate jobs."""
    candidates: list[Job] = []
    # Bound model calls even if it never chooses stop.
    for _ in range(4):
        try:
            name, args = next_action(memory)
            if name == "stop":
                memory.short_term.append(Step(state={}, action=name, args=args))
                break
            if name == "search_jobs":
                found = tools.search_jobs(**args)
                if memory.use_episodic:
                    memory.episodic.queries.append(args["query"])
                    found = [job for job in found if job.job_id not in memory.episodic.seen_job_ids]
                    memory.episodic.seen_job_ids.update(job.job_id for job in found)
                candidates = list({job.job_id: job for job in [*candidates, *found]}.values())
                result = [{"job_id": job.job_id, "title": job.title} for job in found]
            elif name == "filter_jobs":
                candidates = tools.filter_jobs(candidates, **args)
                result = [{"job_id": job.job_id, "title": job.title} for job in candidates]
            else:
                job = next((job for job in candidates if job.job_id == args["job_id"]), None)
                if job is None:
                    raise ValueError("Job is not in the current candidates")
                result = asdict(tools.get_job_details(job.job_id)) if name == "get_job_details" else {"overlap": tools.skill_overlap(memory.semantic, job)}
            memory.short_term.append(Step(state={}, action=name, args=args, result=result))
        except (ValueError, KeyError) as exc:
            action = "stop" if candidates and str(exc) == "Repeated search query" else "error"
            memory.short_term.append(Step(state={}, action=action, args={}, feedback=str(exc)))
            break
    return candidates
