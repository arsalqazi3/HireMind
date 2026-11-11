# HireMind checklist

Last updated: 2026-11-11

Updated after every finished task. Steps follow the proposal (`HireMind_Proposal.pdf`).

## Done

- [x] Project skeleton: folders, stubs, config, tests (`dc7c592`)
- [x] Python 3.13 venv with all requirements
- [x] Kaggle datasets downloaded to `data/raw/`
- [x] Data exploration, written up in `docs/data_notes.md` (`648aa74`)
- [x] Job to resume category mapping rules in `docs/category_mapping.md` (`648aa74`)
- [x] `scripts/clean_data.py`: 2,481 resumes, job pool of 10,013 (`74cd4ea`)
- [x] `scripts/split_data.py`: 1,736 train, 372 validation, 373 test (`74cd4ea`)
- [x] Shared skill list, 418 skills, in `hiremind/skills.csv` (`a9cc080`)
- [x] FAISS index code, ranking metrics (P@5, MRR, NDCG@5) and the FAISS-only baseline, tested with a fake embedder (`7538381`)
- [x] Shared Google Drive folder [HireMind-shared](https://drive.google.com/drive/folders/162kfIrXeEE7Ku_BMO7dpCmVxHOor4Jnb), with Hassaan as editor
- [x] `docs/sharing.md`: how the team shares large files (`2a617f2`)
- [x] Embedding model downloaded (`all-MiniLM-L6-v2`) and job index built: 10,013 jobs, about 15 minutes on CPU
- [x] Index check on validation resumes: 52.8% of the top 5 jobs share the resume's category (random would be about 4%). This is a check, not the proposal's baseline score
- [x] Google Drive for desktop set up. `data/processed/`, `data/splits/` and `models/index/` uploaded to the shared folder, with Drive's checksums matching the local files

- [x] Relevance labelling set: 200 validation pairs, sheets on Drive in `HireMind-shared/labels/`, import script ready (`docs/relevance_labelling.md`)

## In progress

- [ ] Arslan labels `relevance_arslan.xlsx` (115 pairs)
- [ ] Hassaan labels `relevance_hassaan.xlsx` (115 pairs)
- [ ] Relevance rule features (resume years of experience and level, job level, skill overlap), built while labelling happens

## Decisions for the team

- [ ] Hand-check the categories of the 100 test resumes used for matching (recommended)
- [ ] Decide how to fill missing experience levels (empty for 26.8% of the job pool)
- [ ] Hassaan reviews `docs/category_mapping.md` and `hiremind/skills.csv`

## To do

### Matching evaluation setup

- [ ] Relevance rule (category, skills and level), with cut-offs tuned on the hand labels
- [ ] Score ablation A (FAISS only) on the test set
- [ ] TF-IDF plus Logistic Regression baseline for the resume category

### Training data (rule-based labels)

- [ ] Parser labels: category, skills, years of experience, education, roles, projects
- [ ] Critic labels: about 20,000 resume and job pairs with a score out of 10 and a reason
- [ ] Rule-based Searcher agent, and its saved steps as training traces
- [ ] Writer examples from cover letter templates

### Model

- [ ] Pick the base SLM (a few hundred million parameters), checking speed and memory on the laptop CPU
- [ ] Fine-tune all four roles with LoRA on Colab, choosing the checkpoint on validation
- [ ] Merge the LoRA weights if it helps CPU speed

### Agent

- [ ] `hiremind/slm.py`: load the model and generate with task tokens
- [ ] Tools: `search_jobs`, `filter_jobs`, `get_job_details`, `skill_overlap`
- [ ] Roles: Parser, Searcher (ReAct loop, at most 3 rounds), Critic (cutoff 6), Writer with claim checker
- [ ] Memory: semantic, episodic and short-term
- [ ] `hiremind/agent.py`: the full loop with Critic retries

### App

- [ ] Streamlit app: upload a resume, see the profile, watch the steps, get 5 jobs and a cover letter

### Evaluation and report

- [ ] Per-role evaluation on the test set (`evaluation/eval_tasks.py`)
- [ ] Ablation versions B to E
- [ ] Practical checks: CPU time, memory, model size, retries, unsupported claims in cover letters
- [ ] Evaluation report
- [ ] Final report and demo video
