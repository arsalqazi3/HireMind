"""Fine-tune Qwen3-0.6B on the Parser, Critic and Searcher examples with LoRA.

Run on a Colab GPU (see training/train_lora_colab.ipynb):
    python -m training.train_lora --data <folder with the JSONL files> --out <adapter folder>

Each example is formatted exactly like hiremind.slm.generate formats a prompt at
run time (chat template, task token, thinking off), and the loss is computed on
the answer tokens only. The Critic set is sampled because the full set is about
25M tokens, too much for a free Colab session. The checkpoint with the lowest
validation loss is kept, as the proposal says.
"""
import argparse
import json
import random
from pathlib import Path

from hiremind import config

TASKS = ("parser", "critic", "searcher")
BASE_MODEL = "Qwen/Qwen3-0.6B"
# How many examples of each task to use; None means all of them.
TRAIN_LIMITS = {"parser": None, "critic": 6000, "searcher": None}
VAL_LIMITS = {"parser": 150, "critic": 300, "searcher": 300}


def load_examples(data_dir: Path, split: str, limits: dict, seed: int = config.SEED) -> list[dict]:
    """Read <task>_<split>.jsonl for each task and sample down to the limits."""
    rng = random.Random(seed)
    examples = []
    for task in TASKS:
        with (Path(data_dir) / f"{task}_{split}.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        if limits.get(task) is not None and len(rows) > limits[task]:
            rows = rng.sample(rows, limits[task])
        examples += [{"task": r["task"], "prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    rng.shuffle(examples)
    return examples


def encode(example: dict, tokenizer, max_len: int) -> dict | None:
    """Token ids with the prompt masked out of the loss, or None if too long.

    Long examples are dropped, not cut, because cutting would remove the answer.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{config.TASK_TOKENS[example['task']]}\n{example['prompt']}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_len:
        return None
    return {"input_ids": prompt_ids + answer_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
            "labels": [-100] * len(prompt_ids) + answer_ids}


def encode_all(examples: list[dict], tokenizer, max_len: int) -> tuple[list[dict], int]:
    encoded = [encode(e, tokenizer, max_len) for e in examples]
    kept = [e for e in encoded if e is not None]
    return kept, len(encoded) - len(kept)


def collate(batch: list[dict], pad_id: int) -> dict:
    """Pad a batch to its longest example; padding is ignored by the loss."""
    import torch

    width = max(len(e["input_ids"]) for e in batch)
    pad = lambda values, fill: values + [fill] * (width - len(values))
    return {
        "input_ids": torch.tensor([pad(e["input_ids"], pad_id) for e in batch]),
        "attention_mask": torch.tensor([pad(e["attention_mask"], 0) for e in batch]),
        "labels": torch.tensor([pad(e["labels"], -100) for e in batch]),
    }


def train_adapter(model, tokenizer, train: list[dict], val: list[dict], out_dir, epochs: float = 2,
                  resume: bool = False, max_steps: int = -1, eval_steps: int = 200):
    """Wrap `model` with LoRA, train, keep the checkpoint with the lowest
    validation loss, and save that adapter to <out_dir>/final."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    gpu = torch.cuda.is_available()
    bf16 = gpu and torch.cuda.is_bf16_supported()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,  # a fraction of the total steps
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=bf16,
            fp16=gpu and not bf16,
            train_sampling_strategy="group_by_length",
            report_to="none",
            seed=config.SEED,
        ),
        train_dataset=train,
        eval_dataset=val,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=resume or None)
    final = Path(out_dir) / "final"
    trainer.model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    return trainer, final


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--data", required=True, help="folder with <task>_<split>.jsonl")
    arguments.add_argument("--out", required=True, help="where checkpoints and the final adapter are saved")
    arguments.add_argument("--epochs", type=float, default=2)
    arguments.add_argument("--max-len", type=int, default=3072)
    arguments.add_argument("--critic-limit", type=int, default=TRAIN_LIMITS["critic"])
    arguments.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    args = arguments.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train, dropped_train = encode_all(load_examples(args.data, "train", {**TRAIN_LIMITS, "critic": args.critic_limit}), tokenizer, args.max_len)
    val, dropped_val = encode_all(load_examples(args.data, "val", VAL_LIMITS), tokenizer, args.max_len)
    tokens = sum(len(e["input_ids"]) for e in train)
    print(f"train {len(train)} examples ({dropped_train} too long), {tokens / 1e6:.1f}M tokens per epoch; "
          f"validation {len(val)} ({dropped_val} too long)")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    trainer, final = train_adapter(model, tokenizer, train, val, args.out, args.epochs, args.resume)
    print(f"best checkpoint: {trainer.state.best_model_checkpoint}, eval loss {trainer.state.best_metric:.4f}; adapter saved to {final}")


if __name__ == "__main__":
    main()
