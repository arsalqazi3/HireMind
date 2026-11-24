"""Read the filled-in labelling workbooks and save the labels.

Run as: python -m scripts.import_relevance_labels

Reads data/labels/relevance_<name>.xlsx for each labeller and writes
evaluation/labels/relevance_val.csv (pair_id, labeller, relevant, note), which
holds no dataset text and is committed to git. Stops with a list of problems if
any row is unlabelled or has a value other than 1 or 0.

Rows whose note contains "AI draft" are rejected unless --allow-ai-drafts is
given. With the flag they are imported with source "ai_draft" (otherwise
"human"), so every result built on them can say the labels were AI-made.
"""
import argparse
from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.make_relevance_sheet import KEY_FILE, LABELLERS, SHEET_DIR

OUT_FILE = KEY_FILE.parent / "relevance_val.csv"


def read_labels(name: str, path=None) -> pd.DataFrame:
    sheet = pd.read_excel(path or SHEET_DIR / f"relevance_{name}.xlsx", sheet_name="labels", dtype={"relevant": str})
    sheet["relevant"] = sheet["relevant"].fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    sheet["note"] = sheet["note"].astype("string").fillna("")
    sheet["labeller"] = name
    sheet["source"] = sheet["note"].str.contains("AI draft", case=False, regex=False).map({True: "ai_draft", False: "human"})
    return sheet[["pair_id", "labeller", "relevant", "source", "note"]]


def problems(labels: pd.DataFrame, allow_ai_drafts: bool = False) -> list[str]:
    bad = labels[~labels["relevant"].isin(["0", "1"])]
    issues = [f"{r.labeller} {r.pair_id}: {r.relevant or 'empty'}" for r in bad.itertuples()]
    if allow_ai_drafts:
        return issues
    drafts = labels[labels["note"].fillna("").astype(str).str.contains("AI draft", case=False, regex=False)]
    issues.extend(f"{r.labeller} {r.pair_id}: AI draft needs human review" for r in drafts.itertuples())
    return issues


def agreement(labels: pd.DataFrame) -> dict:
    """Agreement on the pairs both people labelled."""
    wide = labels.pivot(index="pair_id", columns="labeller", values="relevant").dropna()
    if wide.shape[1] < 2 or wide.empty:
        return {"pairs": 0}
    a, b = wide.iloc[:, 0].astype(int), wide.iloc[:, 1].astype(int)
    return {"pairs": len(wide), "same_answer": float((a == b).mean()), "cohen_kappa": float(cohen_kappa_score(a, b))}


def assignment_problems(labels: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    """Require exactly one label for every assigned pair and reviewer."""
    expected = {(str(row.pair_id), name) for row in pairs.itertuples() for name in row.labellers.split("|")}
    actual = [(str(row.pair_id), row.labeller) for row in labels.itertuples()]
    duplicates = {item for item, count in Counter(actual).items() if count > 1}
    return ([f"{name} {pair}: duplicate" for pair, name in sorted(duplicates)]
            + [f"{name} {pair}: missing" for pair, name in sorted(expected - set(actual))]
            + [f"{name} {pair}: unexpected" for pair, name in sorted(set(actual) - expected)])


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--allow-ai-drafts", action="store_true",
                           help="import rows marked 'AI draft' (recorded as source ai_draft)")
    args = arguments.parse_args()
    labels = pd.concat([read_labels(name) for name in LABELLERS], ignore_index=True)
    wrong = problems(labels, args.allow_ai_drafts) + assignment_problems(labels, pd.read_csv(KEY_FILE))
    if wrong:
        raise SystemExit("Fix these rows first:\n" + "\n".join(wrong))
    labels["relevant"] = labels["relevant"].astype(int)
    labels.to_csv(OUT_FILE, index=False)
    print(f"saved {len(labels)} labels to {OUT_FILE}; sources: {labels['source'].value_counts().to_dict()}")
    print("relevant share:", labels.groupby("labeller")["relevant"].mean().round(2).to_dict())
    print("agreement:", agreement(labels))


if __name__ == "__main__":
    main()
