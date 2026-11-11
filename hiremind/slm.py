"""Load and run the local language model on the CPU."""
import os
import platform
from functools import lru_cache

from hiremind import config


@lru_cache(maxsize=2)
def load(model_dir: str | None = None):
    """Load the tokenizer and model once and keep them for the whole run."""
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = model_dir or "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source).to("cpu").eval()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return tokenizer, model


def generate(task: str, prompt: str, max_new_tokens: int = 512) -> str:
    """Put the task token from config.TASK_TOKENS in front of `prompt` and generate."""
    import torch

    tokenizer, model = load()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{config.TASK_TOKENS[task]}\n{prompt}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    if inputs.input_ids.shape[1] + max_new_tokens > 16_384:
        raise ValueError("Prompt exceeds the 16K context budget")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
