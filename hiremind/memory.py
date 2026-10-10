"""The three memory types from class."""
from dataclasses import dataclass, field

from hiremind.schemas import Profile, Step


@dataclass
class EpisodicMemory:
    queries: list[str] = field(default_factory=list)
    seen_job_ids: set[str] = field(default_factory=set)
    rejected: dict[str, str] = field(default_factory=dict)  # job_id -> reason
    critic_feedback: list[str] = field(default_factory=list)


@dataclass
class Memory:
    semantic: Profile | None = None  # the parsed profile
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)
    short_term: list[Step] = field(default_factory=list)  # steps of the current run
    use_episodic: bool = True  # False for ablation C
