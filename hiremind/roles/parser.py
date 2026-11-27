"""Parser: resume (PDF or text) to a Profile."""
from collections import Counter
from io import BytesIO
import json
import re
from pathlib import Path

from hiremind import config, slm
from hiremind.schemas import Profile
from hiremind.skills import extract_skills


SECTION_HEADINGS = {
    "profile": ("profile", "summary", "professional summary"),
    "skills": ("skills", "technical skills", "core skills"),
    "experience": ("experience", "work experience", "professional experience", "employment history"),
    "education": ("education", "continuing education"),
    "projects": ("projects", "selected projects", "academic projects", "personal projects"),
    "certifications": ("certifications", "licenses and certifications"),
    "volunteering": ("volunteering", "volunteer experience", "community involvement"),
}
_SECTION_BY_HEADING = {heading: section for section, headings in SECTION_HEADINGS.items() for heading in headings}
_SPORTS_ROLE = re.compile(
    r"\b(?:basketball|soccer|football|volleyball|baseball|softball|swimming|tennis|track and field|sports)\b"
    r"[^,\n]{0,30}\bcoach\b|\bcoach\b[^,\n]{0,25}\b(?:basketball|soccer|football|volleyball|baseball|softball|swimming|tennis)\b",
    re.I,
)
_ENGINEERING_HEADLINE = re.compile(r"\b(?:engineer|engineering intern|engineering technician)$", re.I)


def _heading(line: str) -> str | None:
    return _SECTION_BY_HEADING.get(line.strip().rstrip(":").strip().casefold())


def _join_wrapped_lines(lines: list[str]) -> list[str]:
    items = []
    for line in lines:
        if items and line[0].islower() and not items[-1].endswith((".", "!", "?")):
            items[-1] += " " + line
        else:
            items.append(line)
    return items


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


def _pdf_page_text(page) -> str:
    from pypdf.errors import PyPdfError

    plain = page.extract_text() or ""
    if not plain.strip():
        return plain
    try:
        layout = page.extract_text(extraction_mode="layout") or ""
    except PyPdfError:
        return plain
    rows = layout.splitlines()
    # A sidebar has a recurring right margin and its own section heading.
    # Right-aligned dates alone are part of the original reading order.
    gaps = [{gap.end() for gap in re.finditer(r"(?<=\S) {5,}(?=\S)", row)} for row in rows]
    right_starts = Counter(position for positions in gaps for position in positions)
    for marker, _ in right_starts.most_common():
        # Different font widths can shift the same visible margin slightly.
        near = set(range(marker - 2, marker + 3))
        matches = [i for i, positions in enumerate(gaps) if near & positions]
        if len(matches) < 3:
            continue
        margin = min(near & right_starts.keys())
        right_start = next((i for i, row in enumerate(rows) if _heading(row[margin:])), None)
        if right_start is None:
            continue
        after_heading = [i for i in matches if i >= right_start]
        if len(after_heading) < 3:
            continue
        # A substantial vertical break after the last paired line can end
        # the columns. Keep subsequent full-width sections in page order.
        last_match = after_heading[-1]
        end = next((i for i in range(last_match + 4, len(rows)) if rows[i].strip()
                    and all(not row.strip() for row in rows[i - 3:i])), len(rows))
        header, body, footer = rows[:right_start], rows[right_start:end], rows[end:]
        if any(row[margin - 2:margin].strip() for row in body):
            continue
        left = [row[:margin].strip() for row in body]
        right = [row[margin:].strip() for row in body]
        ordered = "\n".join(line.strip() for line in [*header, *left, *right, *footer] if line.strip())
        if Counter(ordered.split()) == Counter(plain.split()):
            return ordered
    return plain


def resume_text(name: str, content: bytes) -> str:
    """Extract an uploaded résumé in memory, with actionable format errors."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError

        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("This PDF is password protected. Upload an unlocked copy.")
            text = "\n".join(_pdf_page_text(page) for page in reader.pages)
        except PyPdfError as exc:
            raise ValueError("This PDF could not be read. Export a fresh PDF or upload a TXT copy.") from exc
        if not text.strip():
            raise ValueError("This PDF has no extractable text. Upload a text-based PDF or paste the résumé into a TXT file.")
    elif suffix == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("This TXT file could not be read. Save it as UTF-8 and try again.") from exc
    else:
        raise ValueError("Resume must be a PDF or text file")
    if not text.strip():
        raise ValueError("Resume contains no extractable text")
    return text


def read_resume(path: str) -> str:
    """Read a local résumé using the same extraction as the app."""
    file = Path(path)
    return resume_text(file.name, file.read_bytes())


def build_prompt(resume_text: str) -> str:
    """Build the prompt shared by inference and training examples."""
    return ('Extract a resume profile as JSON with category, years_experience, education, roles and projects. '
              'Use null for unknown years and arrays of exact resume phrases for education, roles and projects.\nResume:\n' + resume_text
              + '\nChoose the best category from this list only: ' + ', '.join(config.CATEGORIES) + '.')


def parse(resume_text: str) -> Profile:
    """Ask the SLM for a structured profile and retry invalid JSON once."""
    if not resume_text.strip():
        raise ValueError("Resume text is empty")
    prompt = build_prompt(resume_text)
    for _ in range(2):
        try:
            answer = slm.generate("parser", prompt, max_new_tokens=512).strip()
        except slm.PromptTooLong:
            break
        try:
            data = json.loads(answer.removeprefix("```json").removesuffix("```").strip())
            category = data["category"]
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Invalid category")
            category = category.strip().upper().replace(" ", "-")
            if category not in config.CATEGORIES:
                raise ValueError("Category is outside the supported list")
            return _build_profile(resume_text, category, data)
        except (ValueError, KeyError, TypeError, AttributeError):
            prompt += "\nReturn only a valid JSON object with the requested fields."
    profile = _build_profile(resume_text, config.UNKNOWN_CATEGORY, {})
    if profile.roles or profile.skills:
        return profile
    raise ValueError("Could not establish enough résumé information for a search")


def _build_profile(resume_text: str, category: str, data: dict) -> Profile:
    fields = {}
    for name in ("education", "roles", "projects"):
        values = data.get(name, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            values = []
        fields[name] = [v for v in values if isinstance(v, str) and v.strip() and v.casefold() in resume_text.casefold()]
    # Recover literal lines from clearly labelled sections when the model
    # returns empty or placeholder arrays. Never infer employers or dates.
    sections: dict[str, list[str]] = {}
    active = None
    for line in resume_text.splitlines():
        if section := _heading(line):
            active = sections.setdefault(section, [])
        elif active is not None and line.strip():
            active.append(line.strip())

    if not fields["roles"]:
        lines = sections.get("experience", [])
        fields["roles"] = [line.split(",", 1)[0].split("|", 1)[0].strip()
                           for line in lines if not line.startswith(("-", "•")) and ("," in line or "|" in line)][:4]
    fields["roles"] = [role for role in fields["roles"] if re.search(r"\w", role)]
    if not fields["education"]:
        fields["education"] = [line.lstrip("-• ") for line in sections.get("education", [])[:2]]
    # A verbatim phrase elsewhere in the resume is not necessarily a
    # project (e.g. an employer and dates). Require section evidence.
    project_lines = _join_wrapped_lines(sections.get("projects", []))
    project_section = " ".join(project_lines)
    fields["projects"] = [value for value in fields["projects"] if value.casefold() in project_section.casefold()]
    if not fields["projects"]:
        fields["projects"] = [line.lstrip("-• ") for line in project_lines[:2]]
    # Use an unambiguous, resume-backed occupation when all recognized
    # roles agree. Conflicting roles leave the model category unchanged.
    # Flattened résumés sometimes put their title on the same line as Summary.
    headline = re.match(r"\A\s*([A-Z][A-Z '’&/-]{3,75}?)\s+(?=Summary\b|Professional Summary\b|Profile\b)",
                        resume_text)
    if headline:
        title = headline.group(1).strip()
        if _SPORTS_ROLE.search(title) and not any(_SPORTS_ROLE.search(role) for role in fields["roles"]):
            fields["roles"].insert(0, title)
        elif not fields["roles"] and _ENGINEERING_HEADLINE.search(title):
            fields["roles"].append(title)
    role_categories = set()
    for role in fields["roles"]:
        if re.search(r"\b(?:software|web|backend|front.?end|full.?stack) (?:engineer|developer)\b|\bdata (?:analyst|scientist)\b", role, re.I):
            role_categories.add("INFORMATION-TECHNOLOGY")
        if re.search(r"\b(?:accountant|bookkeeper)\b", role, re.I):
            role_categories.add("ACCOUNTANT")
        if re.search(r"\b(?:registered nurse|licensed practical nurse)\b", role, re.I):
            role_categories.add("HEALTHCARE")
        if _SPORTS_ROLE.search(role):
            role_categories.add("FITNESS")
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
    experience_lines = _join_wrapped_lines(sections.get("experience", []))
    experience_evidence = [line.lstrip("-• ").strip() for line in experience_lines
                           if line.startswith(("-", "•")) and 15 <= len(line.strip()) <= 300][:12]
    early_career = bool(re.search(r"\b(?:recent (?:college |university )?graduate|seeking (?:an? )?(?:entry.level|junior))\b", resume_text, re.I))
    return Profile(category=category, skills=extract_skills(resume_text), years_experience=years,
                   name=extract_name(resume_text), experience_evidence=experience_evidence,
                   early_career=early_career, **fields)
