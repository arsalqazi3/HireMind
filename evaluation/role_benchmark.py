"""Measure frozen role behavior on held-out résumés with explicit weak targets."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

from evaluation import ablation
from hiremind import config, slm, tools
from hiremind.memory import Memory
from hiremind.roles import critic, parser, searcher, writer
from hiremind.skills import extract_skills
from scripts.make_joint_training import profile_target
from scripts.make_pilot_critic_labels import critic_score


def fixed_cases(sample):
    cases = []
    for row in sample:
        profile, _ = profile_target(row)
        jobs = tools.shortlist_jobs(searcher.profile_queries(profile)[0], profile)
        if not jobs:
            continue
        target_job = next(((job, label) for job in jobs if (label := critic_score(profile, job)) is not None), None)
        if target_job:
            job, label = target_job
            cases.append((row, profile, job, label[0]))
    return cases


def evaluate(cases, adapter: bool) -> dict:
    category_ok = parser_complete = first_search = valid_scores = letters = letters_passed = 0
    reasons_with_extra_skill_mentions = letters_without_unexpected_lines = 0
    errors, latencies = [], []
    for row, reference, job, score in cases:
        started = perf_counter()
        try:
            parsed = parser.parse(row["text"])
            parser_complete += 1
            category_ok += int(parsed.category == row["category"])
        except ValueError:
            pass
        try:
            action, _ = searcher.next_action(Memory(semantic=reference))
            first_search += int(action == "search_jobs")
        except ValueError:
            pass
        judged = critic.judge(reference, [job])
        if judged:
            valid_scores += 1
            errors.append(abs(judged[0].score - score))
            known = {s.casefold() for s in [*reference.skills, *job.skills, *extract_skills(job.title + "\n" + job.description)]}
            mentioned = {s.casefold() for s in extract_skills(judged[0].reason)}
            reasons_with_extra_skill_mentions += int(bool(mentioned - known))
        # Use the app's deterministic Writer as the production control.
        letter = (slm.generate("writer", writer.build_prompt(reference, job), max_new_tokens=512)
                  if adapter else writer.write_checked(reference, job))
        if letter.strip():
            letters += 1
            letters_passed += int(letter.strip() == writer.write_checked(reference, job))
            letters_without_unexpected_lines += int(not writer.unsupported_claims(letter, reference, job))
        latencies.append(perf_counter() - started)
    return {"cases": len(cases), "parser_completed": parser_complete, "category_matches": category_ok,
            "valid_first_searches": first_search, "valid_critic_scores": valid_scores,
            "critic_mae_to_weak_labels": round(mean(errors), 3) if errors else None,
            "critic_reasons_with_extra_skill_mentions": reasons_with_extra_skill_mentions,
            "nonempty_letters": letters, "letters_matching_grounded_template": letters_passed,
            "letters_without_unexpected_template_lines": letters_without_unexpected_lines,
            "mean_role_seconds_per_case": round(mean(latencies), 2) if latencies else None}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--adapter", type=Path, required=True)
    cli.add_argument("--limit", type=int, default=12)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.limit < 1 or not (args.adapter / "adapter_config.json").is_file():
        cli.error("Provide a positive case count and a saved adapter")
    import torch

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cases = fixed_cases(ablation.sample_resumes("test", args.limit))
    if not cases:
        cli.error("No held-out cases had sufficient evidence")
    print(f"Prepared {len(cases)} of {args.limit} held-out cases", flush=True)
    result = {"split": "test", "requested_cases": args.limit}
    for name, model in (("production", ""), ("adapter", str(args.adapter.resolve()))):
        with patch.dict(os.environ, {"HIREMIND_ADAPTER_DIR": model,
                                     "HIREMIND_ADAPTER_ROLES": ",".join(config.TASK_TOKENS)}):
            result[name] = evaluate(cases, bool(model))
            ablation.save_json(args.output, result)
            print(name, json.dumps(result[name]), flush=True)


if __name__ == "__main__":
    main()
