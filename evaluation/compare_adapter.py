"""Compare a trained LoRA adapter with the base model on held-out validation examples.

Run as:
    python -m evaluation.compare_adapter --adapter <adapter folder> [--parser 20 --critic 40 --searcher 40]

Uses data/processed/training/<task>_val.jsonl, leaving out the validation examples
that training used to pick the best checkpoint (training.train_lora.VAL_LIMITS),
so the comparison is on examples neither run has seen. Both models answer the
same prompts, formatted as at run time, with greedy decoding. The targets are the
rule-based labels, so this measures how well each model reproduces those rules,
not real-world quality.

Writes reports/adapter_comparison.json and prints a table.
"""
import argparse
import json
import random
import time
from pathlib import Path

from hiremind import config
from training.train_lora import TASKS, VAL_LIMITS, load_examples

MAX_NEW_TOKENS = {"parser": 256, "critic": 128, "searcher": 96}


def parse_json(text: str):
    """The first JSON object in `text`, or None."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def set_f1(predicted, expected) -> float:
    p = {str(x).strip().casefold() for x in predicted or []}
    e = {str(x).strip().casefold() for x in expected or []}
    if not p and not e:
        return 1.0
    overlap = len(p & e)
    return 0.0 if overlap == 0 else 2 * overlap / (len(p) + len(e))


def score(task: str, answer: str, target: str) -> dict:
    """Per-example measurements; missing or invalid answers count as wrong."""
    got, want = parse_json(answer), json.loads(target)
    if task == "parser":
        ok = got is not None
        return {"valid": ok,
                "category": ok and got.get("category") == want["category"],
                "years": ok and got.get("years_experience") == want["years_experience"],
                "roles_f1": set_f1(got.get("roles"), want["roles"]) if ok else 0.0,
                "education_f1": set_f1(got.get("education"), want["education"]) if ok else 0.0}
    if task == "critic":
        try:
            value = float(got["score"])
        except (TypeError, KeyError, ValueError):
            return {"valid": False, "abs_error": None, "within_1": False}
        error = abs(value - float(want["score"]))
        return {"valid": True, "abs_error": error, "within_1": error <= 1}
    ok = got is not None and got.get("tool") in {"search_jobs", "filter_jobs", "stop"}
    return {"valid": ok,
            "tool": ok and got["tool"] == want["tool"],
            "exact": ok and got["tool"] == want["tool"] and got.get("arguments", {}) == want.get("arguments", {})}


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        if key == "abs_error":
            out["mae"] = round(sum(values) / len(values), 3) if values else None
        else:
            out[key] = round(sum(float(v) for v in values) / len(rows), 3)
    return out


def held_out(data_dir: Path, counts: dict, seed: int) -> list[dict]:
    """Validation examples not used for checkpoint selection, sampled per task."""
    used = {e["prompt"] for e in load_examples(data_dir, "val", VAL_LIMITS)}
    rng = random.Random(seed)
    chosen = []
    for task in TASKS:
        with (data_dir / f"{task}_val.jsonl").open(encoding="utf-8") as f:
            rows = [r for r in map(json.loads, f) if r["prompt"] not in used]
        chosen += rng.sample(rows, min(counts[task], len(rows)))
    return chosen


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--adapter", required=True)
    arguments.add_argument("--data", default=str(config.PROCESSED_DIR / "training"))
    arguments.add_argument("--out", default=str(config.ROOT / "reports" / "adapter_comparison.json"))
    for task, n in (("parser", 20), ("critic", 40), ("searcher", 40)):
        arguments.add_argument(f"--{task}", type=int, default=n)
    args = arguments.parse_args()

    import torch
    from hiremind import slm

    examples = held_out(Path(args.data), {t: getattr(args, t) for t in TASKS}, config.SEED + 1)
    tokenizer, model = slm.load(adapter_dir=args.adapter)

    def answer(example):
        text = slm.format_prompt(tokenizer, example["task"], example["prompt"])
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS[example["task"]], do_sample=False)
        return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    results = {"adapter": [], "base": []}
    started = time.time()
    for i, example in enumerate(examples, 1):
        tuned = answer(example)
        with model.disable_adapter():
            base = answer(example)
        for name, text in (("adapter", tuned), ("base", base)):
            results[name].append({"task": example["task"], "answer": text,
                                   **score(example["task"], text, example["completion"])})
        print(f"{i}/{len(examples)} {example['task']} ({time.time() - started:.0f}s)", flush=True)

    summary = {name: {t: summarise([{k: v for k, v in r.items() if k not in ("task", "answer")}
                                    for r in rows if r["task"] == t])
                      for t in TASKS if any(r["task"] == t for r in rows)}
               for name, rows in results.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"adapter_dir": args.adapter, "summary": summary,
                                          "examples": [{"task": e["task"], "target": e["completion"],
                                                        "adapter": a["answer"], "base": b["answer"]}
                                                       for e, a, b in zip(examples, results["adapter"], results["base"])]},
                                         indent=2, ensure_ascii=False), encoding="utf-8")
    for task in TASKS:
        if task in summary["base"]:
            print(f"\n{task}")
            for key in summary["base"][task]:
                print(f"  {key:14} base {summary['base'][task][key]!s:>7}   adapter {summary['adapter'][task][key]!s:>7}")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
