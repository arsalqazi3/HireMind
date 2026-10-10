"""Searcher: ReAct loop that picks tools until it has candidate jobs.

Reads the profile from semantic memory, and past queries and Critic feedback
from episodic memory, so it does not repeat a failed search.
"""
from hiremind.memory import Memory
from hiremind.schemas import Job


def next_action(memory: Memory) -> tuple[str, dict]:
    """Ask the SLM for the next tool name and its arguments, or ("stop", {})."""
    raise NotImplementedError


def search(memory: Memory) -> list[Job]:
    """Run one search round and return the candidate jobs."""
    raise NotImplementedError
