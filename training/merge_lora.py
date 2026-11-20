"""Merge a local LoRA adapter and save a standalone inference checkpoint."""
import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from hiremind import config


def merge(base_model: str | None, adapter_dir: str, out_dir: str) -> None:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(adapter_dir).resolve()
    output = Path(out_dir).resolve()
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("Missing adapter_config.json")
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    source = base_model or PeftConfig.from_pretrained(adapter).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source)
    base = AutoModelForCausalLM.from_pretrained(source)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    with TemporaryDirectory(dir=output.parent, prefix=".merge-") as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        checkpoint.rename(output)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base-model")
    cli.add_argument("--adapter", required=True)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    merge(args.base_model, args.adapter, args.output)
    print(f"Saved merged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
