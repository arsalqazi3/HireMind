"""Data passed between the roles and the tools."""
from dataclasses import dataclass, field


@dataclass
class Profile:
    category: str
    skills: list[str] = field(default_factory=list)
    years_experience: float | None = None
    education: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)


@dataclass
class Job:
    job_id: str
    title: str
    description: str
    category: str
    skills: list[str] = field(default_factory=list)
    level: str | None = None


@dataclass
class Judgement:
    """Critic output for one job."""

    job_id: str
    score: float  # 0 to 10
    reason: str


@dataclass
class Step:
    """One ReAct step of the Searcher. Also the format of the training traces."""

    state: dict
    action: str
    args: dict
    result: list | dict | None = None
    feedback: str | None = None
