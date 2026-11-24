"""Fit an interpretable relevance rule using reviewed validation pairs."""
import argparse
import csv
import json
from pathlib import Path

from evaluation import ablation, features, retrieval
from hiremind import config, matching, tools
from hiremind.roles import parser


def inputs(profile, job) -> dict[str, float]:
    values = features.relevance_features(profile, job)
    overlap = values["skill_overlap"]
    required = matching._requirements(job.description)
    scope = matching.experience_scope(job, required)
    years = None if scope else matching._experience_years(required or job.description)
    available = profile.years_experience
    level = (values["job_level"] or "").casefold()
    return {
        "same_category": float(values["same_category"]),
        "category_known": float(profile.category != config.UNKNOWN_CATEGORY),
        "skill_overlap": overlap if overlap is not None else 0.0,
        "job_skills_known": float(overlap is not None),
        "job_level_known": float(bool(level)),
        "job_entry_level": float(level in {"entry level", "internship"}),
        "job_senior_level": float(level in {"mid-senior level", "director", "executive"}),
        "specialized_duration": float(scope is not None),
        "comparable_years": float(years is not None and available is not None),
        "experience_shortfall": min(max(years - available, 0), 10.0) if years is not None and available is not None else 0.0,
        "senior_title_conflict": float(matching.early_seniority_conflict(profile, job)),
        "credential_gap": float(bool(matching.missing_title_credentials(profile, job))),
    }


def fit(examples: list[tuple[str, dict, bool]]) -> dict:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    groups = [resume_id for resume_id, _, _ in examples]
    labels = [label for _, _, label in examples]
    if len(set(groups)) < 5 or min(labels.count(True), labels.count(False)) < 5:
        raise ValueError("Calibration requires at least five résumés and five reviewed pairs of each class")
    vectorizer = DictVectorizer()
    matrix = vectorizer.fit_transform([row for _, row, _ in examples])
    observed, predicted, held_probabilities = [], [], []

    def select_threshold(targets, probabilities):
        candidates = [step / 20 for step in range(2, 19)]
        return max(candidates, key=lambda threshold: (
            balanced_accuracy_score(targets, [p >= threshold for p in probabilities]), -abs(threshold - 0.5)
        ))

    folds = 0
    for training, held in GroupKFold(n_splits=5).split(matrix, labels, groups):
        targets = [labels[i] for i in training]
        if len(set(targets)) < 2:
            continue
        model = LogisticRegression(random_state=config.SEED, class_weight="balanced")
        model.fit(matrix[training], targets)
        threshold = select_threshold(targets, model.predict_proba(matrix[training])[:, 1])
        probabilities = model.predict_proba(matrix[held])[:, 1].tolist()
        observed.extend(labels[i] for i in held)
        held_probabilities.extend(probabilities)
        predicted.extend(p >= threshold for p in probabilities)
        folds += 1
    model = LogisticRegression(random_state=config.SEED, class_weight="balanced").fit(matrix, labels)
    return {"pairs": len(examples), "resumes": len(set(groups)), "relevant_pairs": labels.count(True),
            "out_of_fold_pairs": len(observed), "folds": folds,
            "threshold": select_threshold(observed, held_probabilities) if len(set(observed)) > 1 else 0.5,
            "grouped_balanced_accuracy": balanced_accuracy_score(observed, predicted) if len(set(observed)) > 1 else None,
            "intercept": float(model.intercept_[0]),
            "coefficients": dict(zip(vectorizer.get_feature_names_out(), model.coef_[0].tolist()))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, help="Optional validation ablation snapshot")
    cli.add_argument("--judgments", type=Path, help="Completed labels.csv from the matching packet")
    cli.add_argument("--allow-ai-drafts", action="store_true", help="also use labels marked 'AI draft'")
    cli.add_argument("--output", type=Path, default=config.ROOT / "outputs/relevance_rule.json")
    args = cli.parse_args()
    validation = set((config.SPLITS_DIR / "val.txt").read_text().splitlines())
    if bool(args.snapshot) != bool(args.judgments):
        cli.error("Supply both --snapshot and --judgments, or neither for the original review set")
    if args.snapshot:
        snapshot = ablation.from_retrieval(json.loads(args.snapshot.read_text(encoding="utf-8")))
        pool = ablation.review_pool(snapshot)
        if set(pool) - validation:
            cli.error("Calibration inputs must belong to the validation split")
        labels = ablation.read_judgments(args.judgments, pool)
    else:
        directory = config.ROOT / "evaluation/labels"
        path = directory / "relevance_val.csv"
        if not path.is_file():
            cli.error("No imported human reviews are available in evaluation/labels/relevance_val.csv")
        labels = retrieval.reviewed_pairs(retrieval.read_rows(directory / "relevance_pairs.csv"), retrieval.read_rows(path),
                                           args.allow_ai_drafts)
    reviewed = {resume_id for resume_id, rows in labels.items() if rows}
    if reviewed - validation:
        cli.error("Calibration inputs must belong to the validation split")
    judgments = [value for resume_id in reviewed for value in labels[resume_id].values()]
    if len(reviewed) < 5 or min(judgments.count(True), judgments.count(False)) < 5:
        cli.error("Calibration requires at least five résumés and five reviewed pairs of each class")
    with (config.PROCESSED_DIR / "resumes.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["id"]: row["text"] for row in csv.DictReader(handle) if row["id"] in reviewed}
    examples = []
    for resume_id in sorted(reviewed):
        try:
            profile = parser.parse(rows[resume_id])
        except ValueError:
            continue
        for job_id, label in labels[resume_id].items():
            examples.append((resume_id, inputs(profile, tools.get_job_details(job_id)), label))
    result = fit(examples)
    ablation.save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
