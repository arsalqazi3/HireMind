"""Inspect base-model output and CPU timings on one validation resume/job pair.

Run: python -m scripts.benchmark_slm
The output contains dataset text and should not be committed.
"""
import argparse
import csv
import json
import os
import resource
import time

from hiremind import config


def validation_example():
    validation_ids = set((config.SPLITS_DIR / "val.txt").read_text().splitlines())
    with (config.ROOT / "evaluation/labels/relevance_pairs.csv").open(newline="", encoding="utf-8") as f:
        pair = next(row for row in csv.DictReader(f) if row["resume_id"] in validation_ids)
    with (config.PROCESSED_DIR / "resumes.csv").open(newline="", encoding="utf-8") as f:
        resume = next(row for row in csv.DictReader(f) if row["id"] == pair["resume_id"])
    with (config.PROCESSED_DIR / "jobs.csv").open(newline="", encoding="utf-8") as f:
        job = next(row for row in csv.DictReader(f) if row["job_id"] == pair["job_id"])
    return resume, job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["parser", "searcher", "critic", "writer"])
    selected_task = parser.parse_args().task
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resume, job = validation_example()
    model_name = "Qwen/Qwen3-0.6B"
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu").eval()
    print(f"Model load: {time.perf_counter() - start:.1f}s; resume {resume['id']}, job {job['job_id']}")

    prompts = {
        "parser": f"Extract the category, skills, years of experience, education, roles and projects as JSON. Use null for unknown facts.\nResume:\n{resume['text']}",
        "searcher": f'Find jobs for this resume. Return only JSON in this exact structure: {{"tool":"search_jobs","arguments":{{"query":"job and skills"}}}}. Replace "job and skills" with a search query of at most 8 words. Do not repeat words or include an explanation.\nResume:\n{resume["text"]}',
        "critic": f"Assess the fit from 0 to 10 as JSON with score, matching evidence, missing requirements and uncertainty. Use only the supplied texts.\nResume:\n{resume['text']}\nJob:\n{job['title']}\n{job['description']}",
        "writer": f"Write a short cover letter for this job using only facts supported by the resume. Do not invent qualifications.\nResume:\n{resume['text']}\nJob:\n{job['title']}\n{job['description']}",
    }
    for task, prompt in prompts.items():
        if selected_task and task != selected_task:
            continue
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt")
        max_new_tokens = 512
        if inputs.input_ids.shape[1] + max_new_tokens > 16_384:
            raise ValueError(f"{task} exceeds the 16K context budget")
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\n--- {task}: {inputs.input_ids.shape[1]} input tokens, {len(generated)} output tokens, {time.perf_counter() - start:.1f}s ---\n{answer}")
        if task != "writer":
            try:
                json.loads(answer.removeprefix("```json").removesuffix("```").strip())
                print("Valid JSON: yes")
            except json.JSONDecodeError:
                print("Valid JSON: no")
    print(f"\nPeak process RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.0f} MiB (macOS)")


if __name__ == "__main__":
    main()
