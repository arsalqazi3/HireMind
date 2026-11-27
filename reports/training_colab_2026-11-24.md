# LoRA training on Colab (2026-11-24)

The Parser, Critic and Searcher adapter for Qwen3-0.6B was trained with `training/train_lora.py` from `training/train_lora_colab.ipynb` on a free Colab T4 GPU. The app uses the grounded cover-letter Writer.

## Setup

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3-0.6B, chat template with thinking off, task tokens `<parse>`, `<search>`, `<critic>` |
| Training data | Rule-based Parser, Critic and Searcher examples from training resumes only (`data/processed/training/`, on Drive) |
| Examples per epoch | 15,286: all Parser and Searcher examples, 6,000 sampled Critic pairs; examples over 3,072 tokens dropped |
| Validation for checkpoint choice | 300 examples (60 Parser, 120 Critic, 120 Searcher) from validation resumes |
| LoRA | r 16, alpha 32, dropout 0.05, on all attention and MLP projections |
| Optimiser | learning rate 2e-4, cosine schedule, 3% warm-up |
| Batch | 1 example per step, 16 steps per update (effective batch 16) |
| Loss | answer tokens only; the output layer only runs on answer positions |
| Precision | fp16 (the T4 has no fast bf16) |
| Schedule | 1 epoch = 956 updates; validation and a checkpoint every 100 updates |

## Run

- Reached update 867 of 956 (91%) in 2 h 47 min, about 11.5 s per update, before the free GPU quota ran out.
- The best checkpoint is update 800. By then the learning rate had fallen to 1.4e-5 (7% of the peak), so the last 156 updates would have changed the weights little.
- The adapter from update 800 is on Drive in `models/slm/qwen3-0.6b-lora/adapter-step800` (40 MB), with its full training log (`trainer_state.json`).

| Update | Training loss (last 20 updates) | Validation loss |
|---:|---:|---:|
| 20 | 1.056 | |
| 100 | 0.124 | 0.128 |
| 200 | 0.072 | 0.096 |
| 300 | 0.052 | 0.067 |
| 400 | 0.035 | 0.056 |
| 500 | 0.032 | 0.075 |
| 600 | 0.024 | 0.038 |
| 700 | 0.019 | 0.032 |
| 800 | 0.023 | **0.029** |

Validation loss fell steadily except for one bump at update 500, and the gap between training and validation loss stayed small, so there is no sign of overfitting within this epoch. The losses measure how closely the model reproduces the rule-based labels, not how good the matches are.

## Earlier attempt (2026-11-23)

The first run, set to 2 epochs, failed for reasons fixed before this run:

- Colab's preinstalled `torchao` was incompatible with `peft`; the notebook now removes it.
- Two examples per step ran out of GPU memory, because the ~152k-word output layer is large for long prompts; now one example per step with 16-step accumulation.
- bf16 was switched on although the T4 only emulates it (69 s per step); now fp16 (17 s per step).
- The run stopped within its first hour, before any checkpoint, when the browser closed. Since then the loss skips the output layer for prompt tokens, training is 1 epoch, and checkpoints are saved every 100 updates.

## First check against the base model

`evaluation/compare_adapter.py` compares the adapter with the base model on validation examples not used to pick the checkpoint. A three-example run on CPU (one per role): the adapter matched the expected answer on all three. The base model gave an unfinished Parser answer, a Critic score 3 points off, and the wrong Searcher tool. The full comparison (20 Parser, 40 Critic, 40 Searcher) runs on the Mac GPU.

## To reproduce

Open the notebook in Colab with a T4 runtime and run all cells. If the session ends, run the resume cell to continue from the last checkpoint.
