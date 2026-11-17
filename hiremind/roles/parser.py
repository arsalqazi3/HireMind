"""Parser: resume (PDF or text) to a Profile."""
import json
import re
from pathlib import Path

from hiremind import config, slm
from hiremind.schemas import Profile
from hiremind.skills import extract_skills


def extract_name(resume_text: str) -> str | None:
    """Find a likely name near the top; leave ambiguous headers for the user."""
    non_name_words = {"resume", "résumé", "curriculum", "vitae", "test", "fictional", "profile", "summary",
                      "engineer", "developer", "analyst", "accountant", "nurse", "manager", "director", "consultant",
                      "designer", "teacher", "professional", "specialist", "experience", "education", "skills"}
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()][:4]
    for line in lines:
        if len(line) > 60 or any(char in line for char in "@|:/0123456789,•—"):
            continue
        parts = line.split()
        if not 2 <= len(parts) <= 4 or any(part.casefold().strip(".-'") in non_name_words for part in parts):
            continue
        if all(part.replace("-", "").replace("'", "").replace(".", "").isalpha() for part in parts) and all(part[0].isupper() for part in parts):
            return line
    return None


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
            # Recover literal lines from clearly labelled sections when the model
            # returns empty or placeholder arrays. Never infer employers or dates.
            def section(name):
                match = re.search(rf"(?im)^\s*{name}\s*$([\s\S]*?)(?=^\s*(?:PROFILE|SKILLS|EXPERIENCE|EDUCATION|PROJECTS)\s*$|\Z)", resume_text)
                return [line.strip() for line in match.group(1).splitlines() if line.strip()] if match else []

            if not fields["roles"]:
                lines = section("EXPERIENCE")
                fields["roles"] = [line.split(",", 1)[0].split("|", 1)[0].strip()
                                   for line in lines if not line.startswith(("-", "•")) and ("," in line or "|" in line)][:4]
            if not fields["education"]:
                fields["education"] = [line.lstrip("-• ") for line in section("EDUCATION")[:2]]
            # A verbatim phrase elsewhere in the resume is not necessarily a
            # project (e.g. an employer and dates). Require section evidence.
            project_section = "\n".join(section("PROJECTS"))
            fields["projects"] = [value for value in fields["projects"] if value.casefold() in project_section.casefold()]
            if not fields["projects"]:
                fields["projects"] = [line.lstrip("-• ") for line in section("PROJECTS")[:2]]
            # Use an unambiguous, resume-backed occupation when all recognized
            # roles agree. Conflicting roles leave the model category unchanged.
            role_categories = set()
            for role in fields["roles"]:
                if re.search(r"\b(?:software|web|backend|front.?end|full.?stack) (?:engineer|developer)\b|\bdata (?:analyst|scientist)\b", role, re.I):
                    role_categories.add("INFORMATION-TECHNOLOGY")
                if re.search(r"\b(?:accountant|bookkeeper)\b", role, re.I):
                    role_categories.add("ACCOUNTANT")
                if re.search(r"\b(?:registered nurse|licensed practical nurse)\b", role, re.I):
                    role_categories.add("HEALTHCARE")
            if len(role_categories) == 1:
                category = role_categories.pop()
            years = data.get("years_experience")
            if isinstance(years, bool) or not isinstance(years, (int, float)) or not 0 <= years <= 80:
                years = None
            elif not re.search(rf"(?<!\d){years:g}\s*(?:\+\s*)?years?\b", resume_text, re.I):
                years = None
            if years is None:
                explicit = re.findall(r"(?<!\d)(\d{1,2})\s*(?:\+\s*)?years?\s+of\s+(?:professional\s+)?experience\b", resume_text, re.I)
                if len(set(explicit)) == 1 and int(explicit[0]) <= 80:
                    years = float(explicit[0])
            experience_evidence = [line.lstrip("-• ").strip() for line in section("EXPERIENCE")
                                   if line.startswith(("-", "•")) and 15 <= len(line.strip()) <= 300][:12]
            early_career = bool(re.search(r"\b(?:recent (?:college |university )?graduate|seeking (?:an? )?(?:entry.level|junior))\b", resume_text, re.I))
            return Profile(category=category, skills=extract_skills(resume_text), years_experience=years,
                           name=extract_name(resume_text), experience_evidence=experience_evidence,
                           early_career=early_career, **fields)
        except (ValueError, KeyError, TypeError, AttributeError):
            prompt += "\nReturn only a valid JSON object with the requested fields."
    raise ValueError("Could not parse a valid resume profile")
