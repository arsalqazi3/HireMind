"""Read the filled-in labelling workbooks and save the labels.

Run as: python -m scripts.import_relevance_labels

Reads data/labels/relevance_<name>.xlsx for each labeller and writes
evaluation/labels/relevance_val.csv (pair_id, labeller, relevant, note), which
holds no dataset text and is committed to git. Stops with a list of problems if
any row is unlabelled or has a value other than 1 or 0.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.make_relevance_sheet import KEY_FILE, LABELLERS, SHEET_DIR

OUT_FILE = KEY_FILE.parent / "relevance_val.csv"


def read_labels(name: str, path=None) -> pd.DataFrame:
    sheet = pd.read_excel(path or SHEET_DIR / f"relevance_{name}.xlsx", sheet_name="labels", dtype={"relevant": str})
    sheet["relevant"] = sheet["relevant"].fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    sheet["note"] = sheet["note"].fillna("")
    sheet["labeller"] = name
    return sheet[["pair_id", "labeller", "relevant", "note"]]


def problems(labels: pd.DataFrame) -> list[str]:
    bad = labels[~labels["relevant"].isin(["0", "1"])]
    return [f"{r.labeller} {r.pair_id}: {r.relevant or 'empty'}" for r in bad.itertuples()]


def agreement(labels: pd.DataFrame) -> dict:
    """Agreement on the pairs both people labelled."""
    wide = labels.pivot(index="pair_id", columns="labeller", values="relevant").dropna()
    if wide.shape[1] < 2 or wide.empty:
        return {"pairs": 0}
    a, b = wide.iloc[:, 0].astype(int), wide.iloc[:, 1].astype(int)
    return {"pairs": len(wide), "same_answer": float((a == b).mean()), "cohen_kappa": float(cohen_kappa_score(a, b))}


def main() -> None:
    labels = pd.concat([read_labels(name) for name in LABELLERS], ignore_index=True)
    wrong = problems(labels)
    if wrong:
        raise SystemExit("Fix these rows first:\n" + "\n".join(wrong))
    labels["relevant"] = labels["relevant"].astype(int)
    labels.to_csv(OUT_FILE, index=False)
    print(f"saved {len(labels)} labels to {OUT_FILE}")
    print("relevant share:", labels.groupby("labeller")["relevant"].mean().round(2).to_dict())
    print("agreement:", agreement(labels))


if __name__ == "__main__":
    main()
