"""Fine-tune one SLM on all four roles together with LoRA.

Runs on a free Colab GPU. Plan:
1. Pick an open-source SLM with a few hundred million parameters.
2. Add the task tokens from hiremind.config.TASK_TOKENS to the tokenizer.
3. Mix the four JSONL training sets. The validation set decides the mix.
4. Train with PEFT LoRA for about 2 to 3 epochs, keep the best checkpoint on validation.
5. Save the adapter to models/.
"""


def train(base_model: str, out_dir: str, epochs: int = 3) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
