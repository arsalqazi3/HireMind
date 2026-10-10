"""The fine-tuned small language model, shared by all four roles.

Loads the base model plus the LoRA adapter (or the merged weights) from
models/ and runs on the CPU.
"""


def load(model_dir: str | None = None):
    """Load the tokenizer and model once and keep them for the whole run."""
    raise NotImplementedError


def generate(task: str, prompt: str, max_new_tokens: int = 512) -> str:
    """Put the task token from config.TASK_TOKENS in front of `prompt` and generate."""
    raise NotImplementedError
