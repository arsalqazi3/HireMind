# HireMind

A self-correcting job-matching agent. It takes an English resume, finds the five
jobs that fit it best with a reason for each, and writes a cover letter for the
one you pick. Everything runs on our own fine-tuned small language model (SLM)
on a laptop CPU, with no external AI API.

AI4015 Agentic Artificial Intelligence, FAST NUCES Karachi.
Muhammad Hassaan (23K-0603) and Arslan Asad (23I-2505).

## How it works

One SLM is fine-tuned with LoRA and used in four roles, each with its own task token:

1. **Parser** turns the resume into a JSON profile.
2. **Searcher** runs a ReAct loop over four tools (`search_jobs`, `filter_jobs`,
   `get_job_details`, `skill_overlap`), at most three search rounds.
3. **Critic** scores each job out of 10. If the best score is under the cutoff
   (starts at 6), it sends feedback to the Searcher.
4. **Writer** writes the cover letter, then a checker removes any claim the resume does not back.

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

Download these into `data/raw/`:

- Resume Dataset (LiveCareer): https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- LinkedIn Job Postings 2023-2024: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

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
