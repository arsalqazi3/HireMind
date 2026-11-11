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
- [x] Qwen3-0.6B selected as the base-model candidate; local CPU benchmark script added (quality and performance still need broader evaluation)
- [x] Local model loader, search tools, bounded Searcher, Parser, Critic, Writer and agent loop implemented; automated tests cover the main paths
- [x] Local search smoke test retrieved candidates from the shared FAISS index; repeated queries are stopped
- [x] Base-model smoke test: Parser and Critic on synthetic examples, and agent ranking with a stubbed search; not a full real-data evaluation
- [x] Relevance feature extraction for category, skill overlap, resume experience and job level; unknown values are kept separate
- [x] TF-IDF plus logistic regression category baseline implemented and measured on validation (`reports/validation_baseline.md`); test evaluation remains

## In progress

- [ ] Arslan labels `relevance_arslan.xlsx` (115 pairs)
- [ ] Hassaan reviews the 115 AI-drafted suggestions in `relevance_hassaan.xlsx` and removes each `AI draft` marker after review
- [ ] Tune a relevance rule using reviewed labels, including how experience relates to job level
- [ ] Expand validation beyond ten Parser examples and one full-agent resume, including human-reviewed recommendation quality and CPU time
- [ ] Improve and independently validate Parser category accuracy; the prompt matched two of five noisy dataset labels on a separate small sample (`reports/validation_baseline.md`)
- [ ] Improve Searcher stop/retry behavior; the base model repeated a query in the real-index smoke test
- [ ] Further reduce and measure real CPU latency; one validation run improved from about 17 minutes to 6 minutes with Apple Silicon threading, under different background activity (`reports/validation_baseline.md`)
- [ ] Evaluate the Critic's reasons and Writer's claim checking; the Writer currently falls back to a restrictive template for unverified drafts

## Decisions for the team

- [ ] Hand-check the categories of the 100 test resumes used for matching (recommended)
- [ ] Decide how to fill missing experience levels (empty for 26.8% of the job pool)
- [ ] Hassaan reviews `docs/category_mapping.md` and `hiremind/skills.csv`

## To do

### Matching evaluation setup

- [ ] Relevance rule (category, skills and level), with cut-offs tuned on the hand labels
- [ ] Score ablation A (FAISS only) on the test set
- [ ] Evaluate the TF-IDF plus Logistic Regression category baseline on the held-out test split after development

### Training data (rule-based labels)

- [ ] Parser labels: category, skills, years of experience, education, roles, projects
- [ ] Critic labels: about 20,000 resume and job pairs with a score out of 10 and a reason
- [ ] Rule-based Searcher agent, and its saved steps as training traces
- [ ] Writer examples from cover letter templates

### Model

- [ ] Confirm Qwen3-0.6B meets accuracy, CPU speed and memory requirements on representative inputs
- [ ] Fine-tune all four roles with LoRA on Colab, choosing the checkpoint on validation
- [ ] Merge the LoRA weights if it helps CPU speed

### Agent

- [ ] Evaluate and improve the implemented roles and agent loop, including three-round retries, claim checks and memory ablations
- [ ] Integrate and test the fine-tuned adapter in the model loader

### App

- [ ] Streamlit app: upload a resume, see the profile, watch the steps, get 5 jobs and a cover letter

### Evaluation and report

- [ ] Per-role evaluation on the test set (`evaluation/eval_tasks.py`)
- [ ] Ablation versions B to E
- [ ] Practical checks: CPU time, memory, model size, retries, unsupported claims in cover letters
- [ ] Evaluation report
- [ ] Final report and demo video
