"""Compare CPU and Apple MPS generation on the same short synthetic prompt.

Run: python -m scripts.benchmark_devices
This is a small diagnostic, not an end-to-end matching benchmark.
"""
import os
import time

from hiremind import config


def main():
    os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Return JSON with a category and skills for a software engineer with Python and SQL."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for device in ["cpu", *(["mps"] if torch.backends.mps.is_available() else [])]:
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).to(device).eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=8, min_new_tokens=8, do_sample=False)
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            model.generate(**inputs, max_new_tokens=64, min_new_tokens=64, do_sample=False)
            if device == "mps":
                torch.mps.synchronize()
        print(f"{device}: {time.perf_counter() - start:.2f}s for 64 generated tokens", flush=True)
        del model, inputs
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
