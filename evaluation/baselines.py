"""Baselines that do not use the agent."""
from hiremind import config
from hiremind.index import JobIndex, load


def tfidf_logreg_category(train_texts: list[str], train_labels: list[str], test_texts: list[str]) -> list[str]:
    """TF-IDF plus Logistic Regression for the resume category."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=config.SEED))
    model.fit(train_texts, train_labels)
    return model.predict(test_texts).tolist() if test_texts else []


def faiss_only(resume_text: str, k: int = config.TOP_K, index: JobIndex | None = None) -> list[str]:
    """Ablation A: embed the resume, return the k nearest job ids, no agent."""
    index = index or load()
    return [job_id for job_id, _ in index.search(resume_text, k)]


def main():
    import argparse
    import csv
    import json
    from pathlib import Path
    from sklearn.metrics import accuracy_score, f1_score
    from time import perf_counter

    cli = argparse.ArgumentParser(description="Evaluate the fixed resume category baseline.")
    cli.add_argument("--split", choices=("val", "test"), default="val")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    train_ids, held_ids = (
        set((config.SPLITS_DIR / f"{name}.txt").read_text().splitlines()) for name in ("train", args.split)
    )
    if train_ids & held_ids:
        raise ValueError("Training and evaluation IDs overlap")
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    training = [row for row in rows if row["id"] in train_ids]
    held = [row for row in rows if row["id"] in held_ids]
    start = perf_counter()
    predicted = tfidf_logreg_category([r["text"] for r in training], [r["category"] for r in training],
                                     [r["text"] for r in held])
    target = [r["category"] for r in held]
    result = {"split": args.split, "training_resumes": len(training), "evaluation_resumes": len(held),
              "accuracy": accuracy_score(target, predicted),
              "macro_f1": f1_score(target, predicted, average="macro", zero_division=0),
              "seconds": round(perf_counter() - start, 3)}
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
