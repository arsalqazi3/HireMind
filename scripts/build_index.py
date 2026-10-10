"""Build the FAISS job index from data/processed/jobs.csv into models/index/.

Run as: python -m scripts.build_index
"""
from hiremind import config, index

if __name__ == "__main__":
    index.build(str(config.PROCESSED_DIR / "jobs.csv"), str(config.MODELS_DIR / "index"))
