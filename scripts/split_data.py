"""Split resumes 70/15/15 into train, validation and test, stratified by category.

Run as: python -m scripts.split_data

Writes resume ids to data/splits/{train,val,test}.txt. Every later script reads
these files, so everything made from one resume stays in that resume's split.
"""


def split(resumes_path: str, out_dir: str, seed: int = 42) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
