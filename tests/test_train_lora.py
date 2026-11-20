import json

import pytest

from training.train_lora import collate, encode, encode_all, load_examples


class FakeTokenizer:
    """One token per character, with a visible chat template."""
    eos_token = "|"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "U:" + messages[0]["content"] + "\nA:"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_is_masked_and_answer_is_learned():
    ids = encode({"task": "critic", "prompt": "p", "completion": '{"s":1}'}, FakeTokenizer(), 100)
    prompt = "U:<critic>\np\nA:"
    assert ids["labels"][:len(prompt)] == [-100] * len(prompt)
    assert "".join(map(chr, ids["labels"][len(prompt):])) == '{"s":1}|'  # answer plus end token
    assert len(ids["input_ids"]) == len(ids["attention_mask"]) == len(ids["labels"])


def test_prompt_uses_the_runtime_task_token():
    ids = encode({"task": "parser", "prompt": "resume", "completion": "{}"}, FakeTokenizer(), 100)
    assert "".join(map(chr, ids["input_ids"])).startswith("U:<parse>\nresume")


def test_too_long_examples_are_dropped_not_cut():
    examples = [{"task": "parser", "prompt": "x" * 50, "completion": "{}"}, {"task": "parser", "prompt": "x", "completion": "{}"}]
    kept, dropped = encode_all(examples, FakeTokenizer(), 30)
    assert dropped == 1 and len(kept) == 1


def test_collate_pads_and_ignores_padding():
    batch = collate([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
                     {"input_ids": [5], "attention_mask": [1], "labels": [5]}], pad_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert batch["attention_mask"].tolist()[1] == [1, 0, 0]
    assert batch["labels"].tolist()[1] == [5, -100, -100]


def test_load_examples_samples_each_task(tmp_path):
    for task in ("parser", "critic", "searcher"):
        with (tmp_path / f"{task}_train.jsonl").open("w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"task": task, "prompt": f"{task}{i}", "completion": "{}", "resume_id": i}) + "\n")
    examples = load_examples(tmp_path, "train", {"parser": None, "critic": 3, "searcher": 5})
    counts = {t: sum(e["task"] == t for e in examples) for t in ("parser", "critic", "searcher")}
    assert counts == {"parser": 10, "critic": 3, "searcher": 5}
    assert set(examples[0]) == {"task", "prompt", "completion"}
    again = load_examples(tmp_path, "train", {"parser": None, "critic": 3, "searcher": 5})
    assert [e["prompt"] for e in again] == [e["prompt"] for e in examples]  # same seed, same sample


def test_lora_training_runs_end_to_end_on_a_tiny_model(tmp_path):
    """Two steps on a tiny random Qwen3 built from a config: no download, CPU only.
    Checks the Trainer, LoRA and saving work together, not training quality."""
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from training.train_lora import train_adapter

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=8))
    tokenizer = FakeTokenizer()
    tokenizer.save_pretrained = lambda path: None
    examples = [{"task": "critic", "prompt": f"profile {i}", "completion": '{"score":5}'} for i in range(8)]
    data, _ = encode_all(examples, tokenizer, 100)
    trainer, final = train_adapter(model, tokenizer, data, data[:2], tmp_path, max_steps=2, eval_steps=1)
    assert trainer.state.global_step == 2
    assert trainer.state.best_metric is not None
    assert (final / "adapter_config.json").exists()
