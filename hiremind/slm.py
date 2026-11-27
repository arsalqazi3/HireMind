"""Load and run the local language model on the available device."""
import os
import platform
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock

from hiremind import config

_MLX_LOCK = Lock()


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


def format_prompt(tokenizer, task: str, prompt: str, system_prompt: str | None = None) -> str:
    messages = [{"role": "user", "content": f"{config.TASK_TOKENS[task]}\n{prompt}"}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@lru_cache(maxsize=2)
def _load_mlx(adapter_dir: str | None):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("The MLX backend requires Apple Silicon macOS")
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    try:
        from mlx_lm import load as mlx_load
    except ImportError as exc:
        raise RuntimeError("Install mlx-lm to use HIREMIND_BACKEND=mlx") from exc
    model, tokenizer = mlx_load(config.BASE_SLM)
    if adapter_dir:
        from hiremind.mlx_adapter import apply_peft_lora

        apply_peft_lora(model, adapter_dir)
    return model, tokenizer


def _generate_mlx(task: str, prompt: str, max_new_tokens: int, system_prompt: str | None,
                  adapter_dir: str | None) -> str:
    # MLX LM shares the tokenizer's streaming detokenizer across generations.
    with _MLX_LOCK:
        model, tokenizer = _load_mlx(adapter_dir)
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        text = format_prompt(tokenizer, task, prompt, system_prompt)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) + max_new_tokens > 16_384:
            raise PromptTooLong("Prompt exceeds the 16K context budget")
        return mlx_generate(model, tokenizer, prompt=tokens, max_tokens=max_new_tokens,
                            sampler=make_sampler(temp=0.0))


def generate(task: str, prompt: str, max_new_tokens: int = 512, *, system_prompt: str | None = None) -> str:
    """Put the task token from config.TASK_TOKENS in front of `prompt` and generate."""
    adapter_dir = os.environ.get("HIREMIND_ADAPTER_DIR")
    selected_roles = {role.strip() for role in os.environ.get("HIREMIND_ADAPTER_ROLES", "parser").split(",")}
    use_adapter = bool(adapter_dir and task in selected_roles)
    backend = os.environ.get("HIREMIND_BACKEND", "pytorch").lower()
    if backend not in {"pytorch", "mlx"}:
        raise ValueError("HIREMIND_BACKEND must be pytorch or mlx")
    device_setting = os.environ.get("HIREMIND_DEVICE", "auto").lower()
    if device_setting not in {"auto", "cpu", "mps"}:
        raise ValueError("HIREMIND_DEVICE must be auto, cpu, or mps")
    if backend == "mlx" and task != "writer" and device_setting != "cpu":
        return _generate_mlx(task, prompt, max_new_tokens, system_prompt,
                             adapter_dir if use_adapter else None)

    import torch

    tokenizer, model = load(adapter_dir=adapter_dir) if use_adapter else load()
    text = format_prompt(tokenizer, task, prompt, system_prompt)
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
            if device != "mps" or device_setting != "auto":
                raise
            warnings.warn(f"MPS generation failed; using CPU: {exc}")
            model.to("cpu")
            tokens = complete()
    return tokenizer.decode(tokens, skip_special_tokens=True)
