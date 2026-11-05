"""Finds skills from the fixed list in hiremind/skills.csv.

The same matching is used on resumes and on job descriptions, so both sides
use the same skill names. Columns of the list:
  terms        matched without regard to case, plurals included
  exact_terms  matched only as written, for words that are risky otherwise
               (for example "Excel" the product and "excel" the verb)
"""
import csv
import re
from functools import lru_cache
from pathlib import Path

SKILLS_FILE = Path(__file__).resolve().parent / "skills.csv"

# Keeps "c++", "c#", "a&p" and "node.js" whole. Hyphens and slashes split words,
# so "cold-calling" matches "cold calling" and "HTML/CSS" gives both skills.
TOKEN = re.compile(r"\.net\b|\.NET\b|[A-Za-z0-9+#&]+(?:\.[A-Za-z0-9+#]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(str(text))


@lru_cache(maxsize=1)
def _load(path: Path = SKILLS_FILE) -> tuple[dict, dict, int]:
    """Return (lowercase terms, exact terms, longest term in words)."""
    loose, exact = {}, {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for term in filter(None, row["terms"].split("|")):
                words = [w.lower() for w in tokenize(term)]
                loose[tuple(words)] = row["skill"]
                loose[(*words[:-1], words[-1] + "s")] = row["skill"]
            for term in filter(None, row["exact_terms"].split("|")):
                exact[tuple(tokenize(term))] = row["skill"]
    longest = max(len(key) for key in [*loose, *exact])
    return loose, exact, longest


def all_skills(path: Path = SKILLS_FILE) -> list[str]:
    with open(path, encoding="utf-8", newline="") as f:
        return [row["skill"] for row in csv.DictReader(f)]


def extract_skills(text: str) -> list[str]:
    """Return the sorted names of the listed skills that appear in `text`."""
    loose, exact, longest = _load()
    tokens = tokenize(text)
    lowered = [t.lower() for t in tokens]
    found = set()
    for i in range(len(tokens)):
        for n in range(1, min(longest, len(tokens) - i) + 1):
            skill = loose.get(tuple(lowered[i:i + n])) or exact.get(tuple(tokens[i:i + n]))
            if skill:
                found.add(skill)
    return sorted(found)
