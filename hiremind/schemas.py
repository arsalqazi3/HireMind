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
    name: str | None = None
    experience_evidence: list[str] = field(default_factory=list)
    early_career: bool = False


@dataclass
class Job:
    job_id: str
    title: str
    description: str
    category: str
    skills: list[str] = field(default_factory=list)
    level: str | None = None
    location: str | None = None
    work_type: str | None = None


@dataclass
class Preferences:
    location: str | None = None
    work_mode: str | None = None
    min_salary: float | None = None
    benefits: str | None = None


@dataclass
class ScoreBreakdown:
    requirements: float | None = None
    experience: float | None = None
    preferences: float | None = None
    qualification: float | None = None
    coverage: float = 0.0
    notes: list[str] = field(default_factory=list)
    mode: str = "standard"
    preference_coverage: float = 0.0


@dataclass
class Judgement:
    """Critic output for one job."""

    job_id: str
    score: float  # 0 to 10
    reason: str
    breakdown: ScoreBreakdown | None = None


@dataclass
class Step:
    """One ReAct step of the Searcher. Also the format of the training traces."""

    state: dict
    action: str
    args: dict
    result: list | dict | None = None
    feedback: str | None = None
