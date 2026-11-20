"""Load and run the local language model on the available device."""
import os
import platform
import warnings
from functools import lru_cache
from pathlib import Path

from hiremind import config


class PromptTooLong(ValueError):
    """The formatted prompt and its answer exceed the model context budget."""


@lru_cache(maxsize=2)
def load(model_dir: str | None = None, adapter_dir: str | None = None):
    """Load the tokenizer and model once and keep them for the whole run."""
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = model_dir or config.BASE_SLM
    if adapter_dir is not None:
        from peft import PeftConfig, PeftModel

        if not (Path(adapter_dir) / "adapter_config.json").is_file():
            raise ValueError("Adapter directory is missing adapter_config.json")
        if model_dir is None:
            source = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path or source
    tokenizer = AutoTokenizer.from_pretrained(source)
    setting = os.environ.get("HIREMIND_DEVICE", "auto").lower()
    if setting not in {"auto", "cpu", "mps"}:
        raise ValueError("HIREMIND_DEVICE must be auto, cpu, or mps")
    if setting == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU is not available on this machine")
    device = "mps" if setting != "cpu" and torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(source)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    try:
        model = model.to(device).eval()
    except (RuntimeError, NotImplementedError) as exc:
        if setting != "auto" or device != "mps":
            raise
        warnings.warn(f"Could not load model on MPS; using CPU: {exc}")
        model = model.to("cpu").eval()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return tokenizer, model


def format_prompt(tokenizer, task: str, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{config.TASK_TOKENS[task]}\n{prompt}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate(task: str, prompt: str, max_new_tokens: int = 512) -> str:
    """Put the task token from config.TASK_TOKENS in front of `prompt` and generate."""
    import torch

    adapter_dir = os.environ.get("HIREMIND_ADAPTER_DIR")
    selected_roles = {role.strip() for role in os.environ.get("HIREMIND_ADAPTER_ROLES", "parser").split(",")}
    tokenizer, model = load(adapter_dir=adapter_dir) if adapter_dir and task in selected_roles else load()
    text = format_prompt(tokenizer, task, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    if inputs.input_ids.shape[1] + max_new_tokens > 16_384:
        raise PromptTooLong("Prompt exceeds the 16K context budget")
    device = model.device.type

    def complete():
        current_inputs = inputs.to(model.device)
        output = model.generate(**current_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # Reading the result can surface an asynchronous GPU error.
        return output[0, current_inputs.input_ids.shape[1]:].tolist()

    with torch.inference_mode():
        try:
            tokens = complete()
        except (RuntimeError, NotImplementedError) as exc:
            if device != "mps" or os.environ.get("HIREMIND_DEVICE", "auto").lower() != "auto":
                raise
            warnings.warn(f"MPS generation failed; using CPU: {exc}")
            model.to("cpu")
            tokens = complete()
    return tokenizer.decode(tokens, skip_special_tokens=True)
