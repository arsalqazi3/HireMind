"""Parser: resume (PDF or text) to a Profile."""
from hiremind.schemas import Profile


def read_resume(path: str) -> str:
    """Return the plain text of a PDF or text resume."""
    raise NotImplementedError


def parse(resume_text: str) -> Profile:
    """Ask the SLM for JSON with category, skills, years of experience,
    education, roles and projects. Retry once if the JSON is not valid."""
    raise NotImplementedError
