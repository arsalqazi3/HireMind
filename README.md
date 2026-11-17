# HireMind

A self-correcting job-matching agent. It takes an English resume, finds up to five
jobs that fit it best with a reason for each, and writes a cover letter for the
one you pick. The prototype uses a local small language model (SLM) and a
structured letter template on a laptop, with no external AI API. On supported Apple Silicon Macs the
model uses the Metal GPU; it falls back to the CPU when unavailable. Set
`HIREMIND_DEVICE=cpu` to force CPU inference.

AI4015 Agentic Artificial Intelligence, FAST NUCES Karachi.
Muhammad Hassaan (23K-0603) and Arslan Asad (23I-2505).

## How it works

The planned system uses one SLM fine-tuned with LoRA across four roles. The
current prototype uses the base SLM for the first three and a structured
template for the Writer; fine-tuning remains future work:

1. **Parser** turns the resume into a JSON profile.
2. **Searcher** runs a ReAct loop over four tools (`search_jobs`, `filter_jobs`,
   `get_job_details`, `skill_overlap`), at most three search rounds.
3. **Critic** scores each job out of 10. If the best score is under the cutoff
   (starts at 6), it sends feedback to the Searcher.
4. **Writer** assembles an editable business-style cover letter from the job
   title and extracted profile facts. The user must fill in contact details
   and review the draft before using it.

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
