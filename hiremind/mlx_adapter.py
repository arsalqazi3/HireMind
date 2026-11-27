"""Load a PEFT Qwen3-0.6B LoRA checkpoint into MLX LM."""

import json
from pathlib import Path
import re

from hiremind import config


_PEFT_WEIGHT = re.compile(r"base_model\.model\.(model\.layers\.\d+\.([\w.]+))\.lora_([AB])\.weight")


def apply_peft_lora(model, adapter_dir: str) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    path = Path(adapter_dir)
    settings = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if (settings.get("base_model_name_or_path") != config.BASE_SLM
            or settings.get("peft_type") != "LORA" or settings.get("bias", "none") != "none"
            or any(settings.get(key) for key in (
                "alpha_pattern", "rank_pattern", "use_dora", "use_rslora", "use_qalora",
                "fan_in_fan_out", "lora_bias", "modules_to_save", "alora_invocation_tokens",
            ))):
        raise ValueError("MLX requires a standard, uniform-rank Qwen3-0.6B PEFT LoRA adapter")

    weights = {}
    target_keys = set()
    for name, weight in mx.load(str(path / "adapter_model.safetensors")).items():
        match = _PEFT_WEIGHT.fullmatch(name)
        if match is None or weight.ndim != 2:
            raise ValueError(f"Unsupported PEFT LoRA tensor for MLX: {name}")
        module, key, factor = match.groups()
        target_keys.add(key)
        # PEFT stores linear weights as (output, input), while MLX LoRA
        # multiplies the two factors on the right of the input activations.
        weights[f"{module}.lora_{factor.lower()}"] = weight.T
    if not weights:
        raise ValueError("The PEFT checkpoint contains no LoRA weights")
    configured_targets = settings.get("target_modules")
    if (not isinstance(configured_targets, list)
            or {key.rsplit(".", 1)[-1] for key in target_keys} != set(configured_targets)):
        raise ValueError("PEFT LoRA weights do not cover the configured target modules")

    rank = settings["r"]
    linear_to_lora_layers(model, len(model.layers), {
        "keys": target_keys, "rank": rank, "scale": settings["lora_alpha"] / rank, "dropout": 0.0,
    })
    parameters = {name: value for name, value in tree_flatten(model.parameters())
                  if name.endswith((".lora_a", ".lora_b"))}
    if parameters.keys() != weights.keys() or any(weights[key].shape != param.shape
                                                   for key, param in parameters.items()):
        raise ValueError("PEFT LoRA tensor coverage or shapes do not match the MLX model")
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())
