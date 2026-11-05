"""Clean both Kaggle datasets.

Run as: python -m scripts.clean_data

Inputs (download into data/raw/):
- resume-dataset (LiveCareer, 2,484 resumes, 24 categories)
- linkedin-job-postings (2023-2024, 123,849 postings)

Outputs in data/processed/:
- resumes.csv: id, category, text
- jobs.csv: about config.JOB_POOL_SIZE postings, balanced across the 24
  categories using the rules in docs/category_mapping.md
"""
import re
from pathlib import Path

import pandas as pd

from hiremind import config

MIN_WORDS = 50

# The rules below are explained in docs/category_mapping.md. Keep the two in step.
EXCLUDE = re.compile(
    r"sandwich artist|state farm|highlands ranch|health (and|&) safety|environmental health"
    r"|pilot flying|pilot travel|pilot company|wind farm|solar farm|farm reporter|night auditor"
    r"|crop insurance|agriculture derivatives|fancy farm|\.net ui|ui tax|materials developer"
    r"|fueler|pizza|store associate|executive assistant|medical drivers",
    re.I,
)

# First match wins, so specific domains come before general ones.
# (?-i:...) is case-sensitive, so "IT" does not match the word "it".
TITLE_RULES = [
    ("INFORMATION-TECHNOLOGY", r"\bsoftware (engineer|developer|development engineer)|\bsalesforce\b|\bcitrix\b|\bvoip\b|\bcisco\b|\bcmdb\b|exchange engineer|system support|microsoft systems|machine learning|\bml engineer|data engineering"),
    ("CHEF", r"\b(chef|sous|cook|culinary|pastry|kitchen manager|baker)\b(?! hughes| tilly)"),
    ("AVIATION", r"\b(aviation|aircraft|airframe|avionics?|a&p|airline|flight attendant|helicopter|pilot)\b(?! plant| program| project)"),
    ("AUTOMOBILE", r"\b(automotive|automobile|auto (body|technician|mechanic|sales|service|parts|detailer)|collision|service advisor|diesel (technician|mechanic)|car (sales|dealer)|vehicle technician|lube tech)\b"),
    ("AGRICULTURE", r"\b(agricultur\w*|agronom\w*|farm|farmer|crop|horticultur\w*|greenhouse|irrigation|soil|livestock|extension agent)\b"),
    ("APPAREL", r"\b(apparel|fashion|garment|textile|footwear|sewing|sewist|seamstress|tailor)\b"),
    ("FITNESS", r"\b(fitness|personal trainer|yoga|pilates|gym|strength (and|&) conditioning|group exercise|wellness coach)\b"),
    ("ADVOCATE", r"\b(advocate|social worker|case worker|caseworker|community outreach)\b"),
    ("BPO", r"\b(call cent(er|re)|contact cent(er|re)|customer (service|support|care)|client support|workforce management)\b|(?-i:\bBPO\b)"),
    ("PUBLIC-RELATIONS", r"\b(public relations|media relations|publicist|communications? (specialist|manager|director|coordinator|officer|associate)|corporate communications?)\b|(?-i:\bPR (professional|assistant|manager|specialist|coordinator|director|intern)\b)"),
    ("DIGITAL-MEDIA", r"\b(digital media|social media|digital marketing|media buyer|paid media|programmatic|content (creator|strategist|producer)|video (producer|editor)|digital content)\b|(?-i:\b(SEO|PPC)\b)"),
    ("ARTS", r"\b(artist|art (director|teacher|supervisor)|museum|curator|gallery|theat(er|re)|musician|illustrator|choreographer|photographer|dance (teacher|instructor)|(performing|visual|fine|language) arts)\b"),
    ("DESIGNER", r"\b(designer|user experience|creative director)\b|(?-i:\b(UX|UI)\b)"),
    ("TEACHER", r"(?<!clinical )(?<!nurse )(?<!informatics )\b(teachers?|educator|teaching|tutor|paraprofessional|professor|lecturer|preschool)\b"),
    ("HR", r"\b(human resources?|hrbp|recruiter|recruiting|talent acquisition|people operations|benefits (specialist|administrator)|compensation analyst)\b|(?-i:(?<![/\d$])\bHR\b)"),
    ("BUSINESS-DEVELOPMENT", r"\b(business development|bdr|partnerships? manager|strategic partnerships?)\b"),
    ("BANKING", r"\b(bank|banker|banking|teller|loan|mortgage|credit analyst)\b"),
    ("ACCOUNTANT", r"\b(accountant|accounting|bookkeep\w*|accounts? (payable|receivable)|auditor|audit|tax|controller)\b|(?-i:\bCPA\b)"),
    ("FINANCE", r"\b(financ\w*|fp&a|treasury|investment|wealth|portfolio manager)\b|(?-i:\bCFO\b)"),
    ("SALES", r"\b(sales|salesperson|account executive|account manager|retail associate|merchandiser|leasing consultant)\b"),
    ("CONSULTANT", r"\bconsult\w*\b"),
    ("INFORMATION-TECHNOLOGY", r"\b(software|developer|programmer|sysadmin|systems? administrator|network|devops|cyber\w*|cloud|help ?desk|database|data (engineer|scientist|analyst)|analytics|information technology|information security|full stack|(front|back) ?end (developer|engineer)|java|python|sql)\b|(?-i:\b(IT|SRE)\b)"),
    ("CONSTRUCTION", r"\b(construction|carpenter|electrician|plumber|superintendent|foreman|estimator|hvac|laborer|mason|roofer|welder|pipefitter)\b"),
    ("HEALTHCARE", r"\b(nurses?|nursing|physician|medical|clinical|health ?care|(mental|behavioral|home|public) health|therapist|pharmacist|pharmacy|dental|dentist|caregiver|surgeon|radiolog\w*|hospital|patient)\b|(?-i:\b(RN|LPN|CNA)\b)"),
    ("ENGINEERING", r"\b(engineer|engineering)\b"),
]
TITLE_RULES = [(category, re.compile(pattern, re.I)) for category, pattern in TITLE_RULES]

APPAREL_INDUSTRY = "Retail Apparel and Fashion"
APPAREL_TITLE = re.compile(r"\b(?:store|retail|sales|selling|merchandis\w*|stylist|buyer|pattern|keyholder|shop|showroom)\b", re.I)


def normalize_text(text: str) -> str:
    """Collapse runs of spaces and blank lines, and keep single line breaks."""
    lines = (re.sub(r"[ \t\r\f\v\xa0]+", " ", line).strip() for line in str(text).split("\n"))
    return "\n".join(line for line in lines if line)


def clean_resumes(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Drop empty, very short and duplicate resumes, and normalize whitespace."""
    df = pd.read_csv(Path(raw_dir) / "resume-dataset" / "Resume" / "Resume.csv", usecols=["ID", "Resume_str", "Category"])
    df = df.rename(columns={"ID": "id", "Resume_str": "text", "Category": "category"})
    df["text"] = df["text"].fillna("").map(normalize_text)
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    df = df.drop_duplicates(subset="text")
    return df[["id", "category", "text"]].sort_values("id").reset_index(drop=True)


def map_job_category(title: str, industries: list[str] = ()) -> str | None:
    """Map a posting to one of the 24 resume categories, or None if no rule matches."""
    if EXCLUDE.search(title):
        return None
    for category, pattern in TITLE_RULES:
        if pattern.search(title):
            return category
    if APPAREL_INDUSTRY in industries and APPAREL_TITLE.search(title):
        return "APPAREL"
    return None


def balance(jobs: pd.DataFrame, size: int = config.JOB_POOL_SIZE, seed: int = config.SEED) -> pd.DataFrame:
    """Sample with the smallest per-category cap that reaches `size` jobs.

    Categories with fewer jobs than the cap keep all of them.
    """
    counts = jobs["category"].value_counts()
    cap = next((c for c in range(1, counts.max() + 1) if counts.clip(upper=c).sum() >= size), counts.max())
    samples = [g.sample(min(cap, len(g)), random_state=seed) for _, g in jobs.groupby("category")]
    return pd.concat(samples)


def _names_by_job(raw: Path, table: str, mapping: str, key: str, name: str) -> pd.Series:
    links = pd.read_csv(raw / "jobs" / table).merge(pd.read_csv(raw / "mappings" / mapping), on=key)
    # Some industry ids have no name in the mapping table.
    return links.dropna(subset=[name]).groupby("job_id")[name].apply(list)


def clean_jobs(raw_dir: Path = config.RAW_DIR, size: int = config.JOB_POOL_SIZE) -> pd.DataFrame:
    """Join postings with their functions and industries, map categories, dedupe, balance.

    List columns are joined with "|" because the industry names contain commas.
    `functions` is LinkedIn's "skills" table, which holds broad job functions and
    not real skills (see docs/data_notes.md). `level` is empty when LinkedIn has none.
    """
    raw = Path(raw_dir) / "linkedin-job-postings"
    columns = ["job_id", "title", "company_name", "location", "description", "formatted_experience_level", "formatted_work_type"]
    df = pd.read_csv(raw / "postings.csv", usecols=columns)
    df = df.rename(columns={"company_name": "company", "formatted_experience_level": "level", "formatted_work_type": "work_type"})
    df = df.dropna(subset=["title", "description"])

    df["title"] = df["title"].str.strip()
    df["description"] = df["description"].map(normalize_text)
    df = df[df["description"].str.split().str.len() >= MIN_WORDS]
    df = df.drop_duplicates(subset="description")

    industries = _names_by_job(raw, "job_industries.csv", "industries.csv", "industry_id", "industry_name")
    functions = _names_by_job(raw, "job_skills.csv", "skills.csv", "skill_abr", "skill_name")
    job_industries = df["job_id"].map(industries).map(lambda v: v if isinstance(v, list) else [])
    job_functions = df["job_id"].map(functions).map(lambda v: v if isinstance(v, list) else [])

    df["category"] = [map_job_category(t, i) for t, i in zip(df["title"], job_industries)]
    df["industries"] = job_industries.map("|".join)
    df["functions"] = job_functions.map("|".join)
    mapped = df.dropna(subset=["category"])

    print(f"postings after cleaning: {len(df)}, mapped to a category: {len(mapped)}")
    pool = balance(mapped, size)
    columns = ["job_id", "title", "company", "location", "category", "level", "work_type", "industries", "functions", "description"]
    return pool[columns].sort_values(["category", "job_id"]).reset_index(drop=True)


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    resumes = clean_resumes()
    resumes.to_csv(config.PROCESSED_DIR / "resumes.csv", index=False)
    print(f"resumes: {len(resumes)}")

    jobs = clean_jobs()
    jobs.to_csv(config.PROCESSED_DIR / "jobs.csv", index=False)
    print(f"job pool: {len(jobs)}")

    table = pd.DataFrame({"resumes": resumes["category"].value_counts(), "jobs": jobs["category"].value_counts()})
    print(table.fillna(0).astype(int).sort_index().to_string())


if __name__ == "__main__":
    main()
