"""Settings shared by the agent, the scripts and the evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"

# Each role gets its own task token at the start of the input.
TASK_TOKENS = {
    "parser": "<parse>",
    "searcher": "<search>",
    "critic": "<critic>",
    "writer": "<write>",
}

MAX_SEARCH_ROUNDS = 3
CRITIC_CUTOFF = 6.0  # out of 10, tuned on the validation set
TOP_K = 5
JOB_POOL_SIZE = 10_000
CATEGORIES = (
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE", "AVIATION", "BANKING", "BPO",
    "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING",
    "FINANCE", "FITNESS", "HEALTHCARE", "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPLIT = (0.70, 0.15, 0.15)  # train, validation, test
SEED = 42
