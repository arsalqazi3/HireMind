"""Split resumes 70/15/15 into train, validation and test, stratified by category.

Run as: python -m scripts.split_data

Writes resume ids to data/splits/{train,val,test}.txt. Every later script reads
these files, so everything made from one resume stays in that resume's split.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from hiremind import config


def split(resumes_path: Path = config.PROCESSED_DIR / "resumes.csv", out_dir: Path = config.SPLITS_DIR, seed: int = config.SEED) -> dict[str, list[int]]:
    resumes = pd.read_csv(resumes_path, usecols=["id", "category"])
    train_share, val_share, test_share = config.SPLIT

    train, rest = train_test_split(resumes, test_size=val_share + test_share, stratify=resumes["category"], random_state=seed)
    val, test = train_test_split(rest, test_size=test_share / (val_share + test_share), stratify=rest["category"], random_state=seed)

    splits = {"train": train, "val": val, "test": test}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(map(str, sorted(part["id"]))) + "\n", encoding="utf-8")

    table = pd.DataFrame({name: part["category"].value_counts() for name, part in splits.items()})
    print(table.fillna(0).astype(int).sort_index().to_string())
    print({name: len(part) for name, part in splits.items()})
    return {name: sorted(part["id"]) for name, part in splits.items()}


if __name__ == "__main__":
    split()
