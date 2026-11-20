"""Compare dense, BM25 and fused retrieval on the existing validation catalogue."""
import argparse
from collections import Counter
from contextlib import closing
from pathlib import Path
import re
import sqlite3
from statistics import median
from time import perf_counter

from evaluation import ablation, retrieval
from hiremind import config, tools
from hiremind.skills import extract_skills

METHODS = ("dense", "bm25", "hybrid")
RANK_CONSTANT = 60


def build_lexical_index(connection, jobs):
    connection.execute(
        "CREATE VIRTUAL TABLE postings USING fts5(job_id UNINDEXED, title, skills, description, tokenize='porter unicode61')"
    )
    connection.executemany("INSERT INTO postings VALUES (?, ?, ?, ?)", (
        (job.job_id, job.title, " ".join(job.skills), job.description) for job in jobs
    ))


def lexical_search(connection, query: str, depth: int) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    terms = Counter(word for word in re.findall(r"[^\W_]+", query.casefold())
                    if len(word) >= 2 and not word.isdecimal() and word not in ENGLISH_STOP_WORDS)
    if not terms:
        return []
    match = " OR ".join(f'"{word}"' for word, _ in terms.most_common(32))
    rows = connection.execute(
        "SELECT job_id FROM postings WHERE postings MATCH ? "
        "ORDER BY bm25(postings, 0.0, 3.0, 2.0, 1.0), job_id LIMIT ?", (match, depth)
    )
    return [job_id for job_id, in rows]


def fuse(*rankings: list[str]) -> list[str]:
    scores = Counter()
    for ranking in rankings:
        for rank, job_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[job_id] += 1 / (RANK_CONSTANT + rank)
    return sorted(scores, key=lambda job_id: (-scores[job_id], job_id))


def compare(resumes, jobs, dense_index, reviewed, depth, query_source="resume"):
    runs = {method: {} for method in METHODS}
    with closing(sqlite3.connect(":memory:")) as connection:
        started = perf_counter()
        build_lexical_index(connection, jobs.values())
        build_seconds = perf_counter() - started
        for resume in resumes:
            query = resume["text"]
            if query_source == "skills":
                query = " ".join(extract_skills(query)) or query
            started = perf_counter()
            dense = [job_id for job_id, _ in dense_index.search(query, depth)]
            dense_seconds = perf_counter() - started
            started = perf_counter()
            lexical = lexical_search(connection, query, depth)
            lexical_seconds = perf_counter() - started
            started = perf_counter()
            hybrid = fuse(dense, lexical)[:depth]
            hybrid_seconds = dense_seconds + lexical_seconds + perf_counter() - started
            for method, ranking, elapsed in zip(METHODS, (dense, lexical, hybrid),
                                                (dense_seconds, lexical_seconds, hybrid_seconds)):
                runs[method][resume["id"]] = {
                    "job_ids": ranking[:config.REVIEW_LIMIT], "seconds": round(elapsed, 6),
                    "category_matches": sum(jobs[job_id].category == resume["category"]
                                            for job_id in ranking[:config.TOP_K]),
                }
    summary = {}
    for method, records in runs.items():
        samples = [(record["job_ids"], reviewed.get(resume_id, {})) for resume_id, record in records.items()]
        summary[method] = {
            "category_matches_at_5": sum(record["category_matches"] for record in records.values()),
            "category_slots_filled_at_5": sum(len(record["job_ids"][:config.TOP_K]) for record in records.values()),
            "median_seconds": median(record["seconds"] for record in records.values()),
            "at_5": retrieval.summarize(samples, config.TOP_K),
            "at_review_limit": retrieval.summarize(samples, config.REVIEW_LIMIT),
        }
    return {"settings": {"split": "val", "resume_ids": [resume["id"] for resume in resumes],
                         "depth": depth, "query_source": query_source},
            "lexical_build_seconds": round(build_seconds, 3), "runs": runs, "summary": summary}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--depth", type=int, default=50)
    cli.add_argument("--query-source", choices=("resume", "skills"), default="resume")
    args = cli.parse_args()
    if args.depth < config.REVIEW_LIMIT:
        cli.error(f"Depth must be at least {config.REVIEW_LIMIT}")
    if args.output.exists():
        cli.error("Choose a new output path to preserve existing results")

    # Load PyTorch before loading FAISS on macOS, as in the ablation runner.
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    from hiremind import index

    ids = set((config.SPLITS_DIR / "val.txt").read_text(encoding="utf-8").splitlines())
    resumes = [row for row in retrieval.read_rows(config.PROCESSED_DIR / "resumes.csv") if row["id"] in ids]
    if not resumes:
        cli.error("The existing validation split has no available résumés")
    label_dir = config.ROOT / "evaluation" / "labels"
    labels = label_dir / "relevance_val.csv"
    reviewed = retrieval.reviewed_pairs(retrieval.read_rows(label_dir / "relevance_pairs.csv"),
                                        retrieval.read_rows(labels)) if labels.is_file() else {}
    result = compare(resumes, tools._jobs(), index.load(), reviewed, args.depth, args.query_source)
    ablation.save_json(args.output, result)
    print(f"Compared {len(resumes)} validation résumés. Saved results to {args.output}.")


if __name__ == "__main__":
    main()
