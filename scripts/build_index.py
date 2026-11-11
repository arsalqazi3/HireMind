"""Build the FAISS job index from data/processed/jobs.csv into models/index/.

Run as: python -m scripts.build_index
"""
from functools import partial

from hiremind import index

if __name__ == "__main__":
    built = index.build(embed_fn=partial(index.embed, show_progress=True))
    print(f"indexed {built.index.ntotal} jobs, {built.index.d} dimensions")
