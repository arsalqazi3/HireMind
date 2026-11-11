"""Parser: resume (PDF or text) to a Profile."""
import json
import re
from pathlib import Path

from hiremind import config, slm
from hiremind.schemas import Profile
from hiremind.skills import extract_skills


def read_resume(path: str) -> str:
    """Return the plain text of a PDF or text resume."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError("Resume must be a PDF or text file")
    if not text.strip():
        raise ValueError("Resume contains no extractable text")
    return text


def parse(resume_text: str) -> Profile:
    """Ask the SLM for JSON with category, skills, years of experience,
    education, roles and projects. Retry once if the JSON is not valid."""
    if not resume_text.strip():
        raise ValueError("Resume text is empty")
    prompt = ('Extract a resume profile as JSON with category, years_experience, education, roles and projects. '
              'Use null for unknown years and arrays of exact resume phrases for education, roles and projects.\nResume:\n' + resume_text
              + '\nChoose the best category from this list only: ' + ', '.join(config.CATEGORIES) + '.')
    for _ in range(2):
        answer = slm.generate("parser", prompt, max_new_tokens=512).strip()
        try:
            data = json.loads(answer.removeprefix("```json").removesuffix("```").strip())
            category = data["category"]
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Invalid category")
            category = category.strip().upper().replace(" ", "-")
            if category not in config.CATEGORIES:
                raise ValueError("Category is outside the supported list")
            fields = {}
            for name in ("education", "roles", "projects"):
                values = data.get(name, [])
                if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                    raise ValueError(f"Invalid {name}")
                fields[name] = [v for v in values if v.strip() and v.casefold() in resume_text.casefold()]
            years = data.get("years_experience")
            if isinstance(years, bool) or not isinstance(years, (int, float)) or not 0 <= years <= 80:
                years = None
            elif not re.search(rf"(?<!\d){years:g}\s*(?:\+\s*)?years?\b", resume_text, re.I):
                years = None
            return Profile(category=category, skills=extract_skills(resume_text), years_experience=years, **fields)
        except (ValueError, KeyError, TypeError, AttributeError):
            prompt += "\nReturn only a valid JSON object with the requested fields."
    raise ValueError("Could not parse a valid resume profile")
