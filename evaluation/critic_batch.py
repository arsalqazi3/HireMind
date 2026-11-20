"""Compare sequential and batched Critic inference without changing production."""
import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

from evaluation import ablation
from hiremind import agent, config, matching, slm, tools
from hiremind.roles import critic, searcher
from scripts.benchmark_slm import peak_rss_mib
from scripts.make_joint_training import profile_target


def generate_batch(prompts, tokenizer, model, max_new_tokens=200):
    import torch

    formatted = [slm.format_prompt(tokenizer, "critic", prompt) for prompt in prompts]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    width = inputs.input_ids.shape[1]
    if width + max_new_tokens > 16_384:
        raise slm.PromptTooLong("Batched Critic input exceeds the 16K context budget")
    with torch.inference_mode():
        decoded = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)[:, width:]
    return tokenizer.batch_decode(decoded, skip_special_tokens=True)


def score(answer: str) -> float | None:
    try:
        row = json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
        value, reason = row["score"], row["reason"]
    except (ValueError, TypeError, KeyError):
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not 0 <= value <= 10 or not isinstance(reason, str)
            or reason.strip().casefold() in {"", "brief evidence and gaps"}):
        return None
    return float(value)


def rank(profile, jobs, scores):
    eligible = []
    for job, value in zip(jobs, scores):
        if value is None or matching.early_seniority_conflict(profile, job):
            continue
        item = matching.score_weighted(profile, job, value)
        if min(item.breakdown.requirements, item.breakdown.qualification) >= config.CRITIC_CUTOFF:
            eligible.append(item)
    return [item.job_id for item in sorted(eligible, key=agent._rank_key, reverse=True)[:config.TOP_K]]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--profiles", type=int, default=2)
    cli.add_argument("--batch-size", type=int, default=4)
    cli.add_argument("--dtype", choices=("checkpoint", "float32"), default="checkpoint")
    args = cli.parse_args()
    if args.output.exists() or args.profiles < 1 or not 2 <= args.batch_size <= config.REVIEW_LIMIT:
        cli.error("Use a fresh output path, at least one profile, and a batch size between 2 and the review limit")

    os.environ.update(HIREMIND_DEVICE="cpu", HIREMIND_ADAPTER_DIR="")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tokenizer, model = slm.load()
    if args.dtype == "float32":
        model.float()
    records = []
    summary = {"device": "cpu", "model": config.BASE_SLM, "dtype": str(model.dtype),
               "batch_size": args.batch_size, "runs": records}
    for row in ablation.sample_resumes("val", args.profiles):
        profile, _ = profile_target(row)
        query = searcher.profile_queries(profile)[0]
        jobs = tools.shortlist_jobs(query, profile)[:args.batch_size]
        if len(jobs) < args.batch_size:
            print(f"Skipped résumé {row['id']}: fewer than {args.batch_size} available jobs", flush=True)
            continue
        prompts = [critic.build_prompt(profile, job) for job in jobs]
        if not records:
            print(f"Warming {model.dtype} model on {len(prompts)} fixed jobs", flush=True)
            slm.generate("critic", prompts[0], max_new_tokens=8)
            generate_batch(prompts, tokenizer, model, max_new_tokens=8)
        responses, durations = {}, {}
        order = ("serial", "batch") if len(records) % 2 == 0 else ("batch", "serial")
        for method in order:
            print(f"Résumé {row['id']}: {method} inference on {len(prompts)} jobs", flush=True)
            started = perf_counter()
            if method == "serial":
                responses[method] = [slm.generate("critic", prompt, max_new_tokens=200) for prompt in prompts]
            else:
                responses[method] = generate_batch(prompts, tokenizer, model)
            durations[method] = round(perf_counter() - started, 3)
        scores = {method: [score(answer) for answer in answers] for method, answers in responses.items()}
        records.append({"resume_id": row["id"], "job_ids": [job.job_id for job in jobs], "order": order,
                        "seconds": durations, "scores": scores,
                        "same_output": [a.strip() == b.strip() for a, b in zip(responses["serial"], responses["batch"])],
                        "rankings": {method: rank(profile, jobs, values) for method, values in scores.items()},
                        "cumulative_process_peak_mib": round(peak_rss_mib(), 1)})
        ablation.save_json(args.output, summary)
    if not records:
        cli.error("No complete Critic batches were available")
    print(f"Compared {len(records)} fixed Critic batches. Saved results to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
