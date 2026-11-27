# HireMind

A self-correcting job-matching agent. It takes an English resume, finds up to five
jobs that fit it best with a reason for each, and writes a cover letter for the
one you pick. The prototype uses a local small language model (SLM) on a
laptop, with no external AI API. On supported Apple Silicon Macs the
model uses the Metal GPU; it falls back to the CPU when unavailable. Set
`HIREMIND_DEVICE=cpu` to force CPU inference.

AI4015 Agentic Artificial Intelligence, FAST NUCES Karachi.
Muhammad Hassaan (23K-0603) and Arslan Asad (23I-2505).

[Course project report](reports/final_project.md) · [Live demo runbook](docs/demo.md) ·
[Full evaluation](reports/evaluation_2026-11-18.md)

## How it works

The current prototype uses the local base model across four roles. Dedicated
and mixed LoRA adapters are available for experiments:

1. **Parser** turns the resume into a JSON profile. If the model repeatedly
   returns unusable output, the Parser can continue with résumé evidence
   from the rules and leave the category unknown.
2. **Searcher** runs a ReAct loop over four tools (`search_jobs`, `filter_jobs`,
   `get_job_details`, `skill_overlap`), at most three search rounds.
   Each query retrieves up to 100 semantically similar jobs before shortlisting
   up to 20 using résumé role titles, skills and category evidence. Known
   seniority and credential conflicts can be screened before model scoring.
3. **Critic** scores each job out of 10. If the best score is under the cutoff
   (starts at 6), it sends feedback to the Searcher.
4. **Writer** loads [its cover-letter skill](hiremind/skills/cover-letter/SKILL.md)
   as system guidance. The Writer uses Qwen3-8B when it is already available
   through local Ollama. It writes a role-specific opening and closing around
   a résumé example, then checks the claims in a second pass. If Ollama is unavailable,
   it tries the project's smaller model. A draft that fails the checks falls back to
   source text. The applicant reviews and edits the draft before sending.

Memory: semantic (the profile), episodic (queries tried, job ids seen, rejection
reasons, Critic feedback) and short-term (the current run).

## Folders

| Folder | What it is |
|---|---|
| `hiremind/` | The agent: roles, tools, memory, SLM loader, FAISS index |
| `scripts/` | Data cleaning, splits, rule-based labels, Searcher traces, index build |
| `training/` | LoRA fine-tuning (Colab) and merging the weights |
| `evaluation/` | Per-role metrics, baselines, matching metrics and the ablation study |
| `app/` | Streamlit demo |
| `docs/` | Category mapping rules and notes |
| `data/` | Kaggle data and generated files (git ignored) |
| `models/` | Fine-tuned weights and the FAISS index (git ignored) |
| `reports/` | Evaluation results |
| `tests/` | Pytest tests |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

The Writer looks for `qwen3:8b` on the local Ollama server. Set
`HIREMIND_WRITER_OLLAMA_MODEL` to a different installed model tag, or set it to an
empty string to use the project's Qwen3-0.6B model directly.

### Optional MLX matching on Apple Silicon

PyTorch is the default. To try MLX with the same Qwen3-0.6B base weights, install
MLX LM in the project environment and opt in when starting Streamlit.

```bash
python -m pip install mlx-lm
HIREMIND_BACKEND=mlx streamlit run app/streamlit_app.py
```

MLX runs the Parser, Searcher and Critic with the base weights or the step-800
LoRA adapter. To run step 800 on all three matching roles, use the existing PEFT
checkpoint from the shared project folder:

```bash
HIREMIND_BACKEND=mlx \
HIREMIND_ADAPTER_DIR="HireMind-shared/models/slm/qwen3-0.6b-lora/adapter-step800" \
HIREMIND_ADAPTER_ROLES=parser,searcher,critic \
streamlit run app/streamlit_app.py
```

The Writer's small-model fallback and explicit CPU runs use PyTorch. Greedy
generation can return different wording, search queries or rankings across
backends. [M1 Pro benchmark](reports/mlx_benchmark.md)

Download these into `data/raw/`:

- Resume Dataset (LiveCareer): https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- LinkedIn Job Postings 2023-2024: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

To repeat the fictional-resume evaluation, add local `sample_*.txt` files or a
`test_resume.txt` to `data/samples/`, then run `python -m scripts.evaluate_samples
--runs 5`. Evaluation outputs and resumes must remain local.

## Match scores

The provisional overall score uses 60% job requirements, 30% stated experience
and seniority, and 10% user preferences. The app displays the individual
scores, qualification-only score, and assessed weight. Unknown components are
excluded and the remaining weights are rescaled, so a partial score is less
informative than one with full coverage. Candidates must meet the provisional
requirements and qualification cutoffs before preferences affect their ranking.
When fewer than five jobs qualify, the agent may search additional rounds. For
an explicitly early-career search, roles titled or explicitly described as senior are excluded unless the
posting describes a junior role; titles can be wrong, so this is a heuristic.
Some clearly named core tools in job titles or explicit requirements are also
checked against the extracted profile. The tool list and wording detection are
incomplete; missing from a resume does not prove the applicant lacks a skill.
The matcher checks a small set of clear credential titles, including CPA, RN,
LPN/LVN and CNA, for corresponding résumé evidence. It cannot verify licensing.

The requirements score starts with the local model's assessment and is adjusted
using detected skills and category evidence. Experience uses stated durations
when comparable, with an explicitly labelled early-career seniority heuristic.
Location uses a simple text comparison. Work arrangement is scored only when
explicit in the job title or location; eligibility still requires review. The
current processed job data has no structured salary or benefits, so those are
never inferred or scored. When only some specified preferences are assessable,
only the corresponding fraction of the preference weight is used.

In-depth review optionally adds focused model assessments for requirements,
experience, and preferences when supporting data exists. Invalid focused output
falls back to the available standard assessment. These model and rule scores are
experimental, not calibrated probabilities, and the historical postings,
eligibility and qualifications require human review.

## Validate retrieval and try the Parser adapter

Compare dense FAISS search with the new shortlist using the same query per
validation résumé:

```bash
python -m evaluation.retrieval
```

The runner writes IDs and summary metrics to `outputs/retrieval_validation.json`.
It calculates judgment coverage and precision bounds once reviewed labels have
been imported. Category agreement is logged as a separate diagnostic.

The LoRA trainer accepts separate JSONL files with `task`, `prompt` and
`completion` fields. The following pilot uses category labels from the current
dataset and trains only the beginning of the Parser's answer:

```bash
python -m scripts.make_parser_pilot
python -m training.train_pilot_lora --train outputs/parser_lora/train.jsonl \
  --validation outputs/parser_lora/val.jsonl --output models/parser-lora-pilot \
  --epochs 1 --max-steps 40 --max-tokens 1280
python -m scripts.validate_parser --adapter models/parser-lora-pilot \
  --exclude-ids outputs/parser_lora/used_ids.txt
```

To try a compatible adapter in the app, set `HIREMIND_ADAPTER_DIR` to its
directory. The Parser is the default adapter target. Set
`HIREMIND_ADAPTER_ROLES` to a comma-separated list for a multi-role adapter.
Run `python -m scripts.make_pilot_searcher_traces` to generate separate training and
validation examples for Searcher tool selection using the existing job index.
Train on those JSONL files with the same LoRA command, then run
`python -m evaluation.eval_tasks --adapter models/searcher-lora-pilot --exclude-traces outputs/searcher_traces`.
Run `python -m scripts.make_pilot_critic_labels` to build weak Critic training labels from
the existing résumé splits and job index. These labels are for training.
Use reviewed labels for ranking evaluation.

Build and evaluate a mixed pilot after generating Searcher and Critic examples:

```bash
python -m scripts.make_joint_training --train-per-role 16 --val-per-role 6
python -m training.train_pilot_lora --train outputs/joint_training/train.jsonl \
  --validation outputs/joint_training/val.jsonl --output models/joint-lora-pilot \
  --epochs 2 --max-steps 64 --max-tokens 1280
python -m evaluation.role_benchmark --adapter models/joint-lora-pilot \
  --output outputs/joint_heldout_roles.json
```

The role benchmark compares a free-writing adapter with the original
deterministic template. The live app uses the validated evidence-selection
workflow described above.

### Matching ablations

Collect the same résumé cohort for every version. Each completed résumé is
checkpointed, so repeating a command continues from the saved results.

```bash
python -m evaluation.ablation collect --split val --limit 3 --versions A B C D E \
  --adapter models/joint-lora-pilot --output outputs/ablation_val.json
python -m evaluation.ablation packet outputs/ablation_val.json outputs/ablation_review
python -m evaluation.ablation score outputs/ablation_val.json outputs/ablation_review/labels.csv
```

Exact ranking metrics use the common fully judged cohort. The packet and its
source text stay local. See `docs/relevance_labelling.md` for the review criteria.

For a 100-résumé held-out run, `scripts.run_ablation_study` runs each variant
in a fresh process and shares a single time budget:

```bash
python -m scripts.run_ablation_study --adapter models/joint-lora-pilot \
  --output outputs/ablation_test_joint.json --packet outputs/ablation_test_joint_review --hours 5
python -m evaluation.ablation status outputs/ablation_test_joint.json
python -m evaluation.ablation diagnostics outputs/ablation_test_joint.json
```

Reuse the snapshot path to resume a run. Use a new packet directory when
exporting again so existing judgments are preserved.
Diagnostics include result counts, success and retry counts, median and
90th-percentile times, and category agreement on shared completed cases.
See `docs/relevance_labelling.md` to freeze a smaller cohort for blind review.

## Pipeline

```bash
python -m scripts.clean_data
python -m scripts.split_data
python -m scripts.make_labels
python -m scripts.make_searcher_traces
python -m scripts.build_index
# train on Colab with training/train_lora.py, put the weights in models/
python -m evaluation.eval_tasks
python -m evaluation.ablation
streamlit run app/streamlit_app.py
```

## Rules

- Everything made from one resume stays in that resume's split. Never train on the test split.
- Never commit data, model weights or the FAISS index.
