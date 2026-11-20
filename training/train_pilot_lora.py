"""Train a Qwen LoRA adapter on prompt/completion JSONL.

Each row has task, prompt and completion strings. Use separate training and
validation files. Only completion tokens contribute to the training loss.
"""
import argparse
import json
import os
import random
from pathlib import Path

from hiremind import config, slm

def load_examples(path, tokenizer, max_tokens):
    examples = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("task") not in config.TASK_TOKENS
                    or not all(isinstance(row.get(key), str) and row[key].strip()
                               for key in ("prompt", "completion"))):
                raise ValueError(f"Invalid task, prompt or completion in {path} at line {number}")
            prompt = slm.format_prompt(tokenizer, row["task"], row["prompt"])
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            answer = tokenizer.encode(row["completion"], add_special_tokens=False)
            if not row.get("partial", False):
                answer.append(tokenizer.eos_token_id)
            if len(prefix) + len(answer) > max_tokens or not answer:
                skipped += 1
                continue
            examples.append((prefix + answer, [-100] * len(prefix) + answer))
    print(f"{path.name}: {len(examples)} examples, {skipped} over the token budget", flush=True)
    if not examples:
        raise ValueError(f"No usable examples in {path}")
    return examples


def train(args):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.train.resolve() == args.validation.resolve():
        raise ValueError("Training and validation require separate files")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new or empty adapter directory")
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    torch.manual_seed(config.SEED)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu" or device == "mps":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_rows = load_examples(args.train, tokenizer, args.max_tokens)
    val_rows = load_examples(args.validation, tokenizer, args.max_tokens)
    def prompt_tokens(row):
        tokens, labels = row
        return tuple(tokens[:labels.count(-100)])

    training_prompts = {prompt_tokens(row) for row in train_rows}
    if any(prompt_tokens(row) in training_prompts for row in val_rows):
        raise ValueError("A training prompt also occurs in validation")
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    )).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    rng = random.Random(config.SEED)

    def loss_for(row):
        ids, labels = row
        answer_length = len(labels) - labels.count(-100)
        inputs = torch.tensor([ids[:-1]], device=device)
        targets = torch.tensor(ids[-answer_length:], device=device)
        # The preceding position predicts each completion token. Skipping
        # vocabulary logits for the prompt saves memory on long résumés.
        logits = model(input_ids=inputs, logits_to_keep=answer_length, use_cache=False).logits
        return torch.nn.functional.cross_entropy(logits[0].float(), targets)

    best_loss = float("inf")
    steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_rows)
        for row in train_rows:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(row)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite")
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 10 == 0:
                print(f"Step {steps}, training loss {loss.item():.4f}", flush=True)
            if args.max_steps is not None and steps >= args.max_steps:
                break
        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_for(row).item() for row in val_rows) / len(val_rows)
        if not val_loss < float("inf"):
            raise RuntimeError("Validation loss became non-finite")
        print(f"Epoch {epoch}, steps {steps}, validation loss {val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(args.output)
        if args.max_steps is not None and steps >= args.max_steps:
            break
    print(f"Saved best adapter from {steps} training steps to {args.output}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--train", required=True, type=Path)
    cli.add_argument("--validation", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--base-model", default=config.BASE_SLM)
    cli.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    cli.add_argument("--epochs", type=int, default=3)
    cli.add_argument("--max-steps", type=int)
    cli.add_argument("--max-tokens", type=int, default=2048)
    cli.add_argument("--learning-rate", type=float, default=2e-4)
    args = cli.parse_args()
    if args.epochs < 1 or args.max_tokens < 1 or (args.max_steps is not None and args.max_steps < 1):
        cli.error("Epochs, token budget and optional step limit must be positive")
    train(args)


if __name__ == "__main__":
    main()
