# MLX benchmark on Apple M1 Pro

Compared MLX LM 0.32.2 and PyTorch 2.15.0 on macOS 27.2.1 using the same local Qwen3-0.6B base weights, with no adapter or extra quantization. Both backends ran greedily. The machine also had the independent Ollama review running, so these timings describe performance under shared load.

## Single-prompt comparison

Each backend ran the same Parser, Searcher and Critic prompts twice in a fresh process. The table shows the second call for each role.

| Role | PyTorch MPS | MLX | Output comparison |
|---|---:|---:|---|
| Parser | 2.43 s | 2.54 s | Identical text |
| Searcher | 0.38 s | 0.37 s | Same tool, different query terms |
| Critic | 3.37 s | 1.88 s | Same score, different explanation wording |

Peak process RSS was 1,007 MiB on PyTorch and 1,056 MiB on MLX in this probe. Separate allocator readings use backend-specific definitions. The probe used unquantized weights and gives no evidence of a memory saving.

## Full matching workflow

Ran the real `agent.run` entry point with the local catalogue and FAISS index, standard review, and default preferences. Each backend ran the two fictional inputs in the same order in a separate process. The first case includes model startup. Cover-letter generation is outside this timing.

| Résumé | PyTorch MPS | MLX | Result |
|---|---:|---:|---|
| Data analyst | 155.77 s | 68.32 s | Same five ordered jobs and scores |
| Registered nurse | 136.19 s | 63.09 s | Same five ordered jobs and scores |

MLX cut total matching time from 291.96 to 131.41 seconds, a 55% reduction across these two runs. The backends chose the same categories and search queries for both résumés. Peak process RSS was 1,096 MiB for PyTorch and 1,136 MiB for MLX. The two-case comparison supports offering MLX as an option. The change in the standalone Searcher query warrants a larger comparison before changing the default.

The local probe, matching runner and raw measurements are under `outputs/mlx_benchmark/`.

## Step-800 LoRA on MLX

HireMind loads the original PEFT checkpoint on MLX by transposing its 392 LoRA
tensors, checking their coverage and shapes against the model, and applying
the original scale of 32/16. The on-disk checkpoint is read directly.

On 12 validation prompts excluded from checkpoint selection, four per role,
MLX and PyTorch returned identical text for every prompt. Both matched 4/4
Parser categories, got all four Critic scores within one point of the rule
labels, and matched all four Searcher tools and arguments. Across the 12
calls, including model startup on the first call, MLX took 27.43 seconds and
PyTorch MPS took 52.50 seconds. This measures component inference on a small
sample while the local review was running. The raw results are in
`outputs/mlx_benchmark/adapter-mlx.json` and `adapter-pytorch.json`.

The full matching workflow with MLX and step 800 enabled for all three roles
returned five jobs for the fictional data analyst résumé in 55.03 seconds,
with zero retries. This was a smoke test of adapter integration.
