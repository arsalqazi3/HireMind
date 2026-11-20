# HireMind checklist

Last updated: 2026-11-20

Updated after every finished task. Steps follow the proposal (`HireMind_Proposal.pdf`).

## Done

- [x] Project skeleton: folders, stubs, config, tests (`dc7c592`)
- [x] Local Python environment and project dependencies installed
- [x] Kaggle datasets acquired for the original data-processing run; raw data is not included in Git
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
- [x] Added candidate-review progress and cover-letter drafting progress, evidence-based score adjustments, and a repeated-query fallback; these heuristics need validation
- [x] Streamlit demo connects resume upload, matching results, recorded steps and cover letter generation to the local agent; interaction tested with a stubbed model
- [x] `scripts/benchmark_slm.py` reports peak memory on Windows as well as macOS and Linux, so the full test suite runs on both laptops
- [x] Apple Silicon GPU inference with CPU fallback and a device benchmark script
- [x] Weighted matching: job requirements, experience and user preferences, with an optional in-depth review; unavailable preferences are excluded
- [x] Structured, editable cover letters use extracted applicant details and supported resume history
- [x] Updated the Streamlit interface using the supplied design reference, including matching and letter progress indicators
- [x] Ran five matching passes for each of five fictional resumes; the historical findings are in `reports/sample_qualification_2026-11-17.md`

- [x] AI-drafted labels for all 115 pairs in `relevance_arslan.xlsx` (48 relevant, 67 not, 24 marked borderline), each with a reason, marked `AI draft`

- [x] Switched inference from CPU to GPU for speed (Apple Silicon GPU on Hassaan's Mac, Colab GPU for training); CPU is no longer a target. The report must explain this change from the proposal
- [x] Parser and Critic prompts moved into `build_prompt` functions shared by the roles and the training data, so training matches inference
- [x] Rule-based training data (`scripts/make_labels.py`), training and validation resumes only: 1,736 + 372 Parser examples and 20,832 + 2,232 Critic pairs (4 nearest, 4 same-category, 4 other-category jobs per resume). Uploaded to Drive `data/processed/training/`

- [x] Searcher prompt moved into `searcher.build_prompt`; rule-based Searcher traces (`scripts/make_searcher_traces.py`): 5,259 training and 1,084 validation decisions (search, filter, stop), 19% of resumes needing more than one round. Uploaded to Drive
- [x] LoRA training script (`training/train_lora.py`) and Colab notebook (`training/train_lora_colab.ipynb`): formats examples exactly like inference, masks the prompt from the loss, samples 6,000 Critic pairs, keeps the checkpoint with the lowest validation loss. Tested end to end on a tiny random model on CPU; not yet run on Colab

## In progress

- [ ] Hassaan reviews the 115 AI-drafted suggestions in `relevance_hassaan.xlsx` and removes each `AI draft` marker after review
- [ ] Arslan reviews the 115 AI-drafted labels in `relevance_arslan.xlsx` and removes each `AI draft` marker after review
- [ ] Tune a relevance rule using reviewed labels, including how experience relates to job level
- [ ] Expand validation beyond ten Parser examples and one full-agent resume, including human-reviewed recommendation quality and CPU time
- [ ] Improve and independently validate Parser category accuracy; the prompt matched two of five noisy dataset labels on a separate small sample (`reports/validation_baseline.md`)
- [ ] Improve Searcher stop/retry behavior; the base model repeated a query in the real-index smoke test
- [ ] Measure GPU latency per resume on a controlled run (earlier CPU runs: about 17 minutes, then 6 minutes; Mac GPU sample runs: 50-111 seconds)
- [ ] Evaluate the Critic's reasons and the Writer's structured, profile-grounded drafts with human reviewers
- [ ] Calibrate the revised weighted scores and in-depth review against human-reviewed matches; the earlier fictional-resume runs predate these changes

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

- [ ] Improve Parser labels: years are found for only 12% of resumes (stated years only), roles for 75%, education for 70%; projects are always empty
- [ ] Decide the Critic label weights with the reviewed relevance labels (now 4 points for same category, up to 6 for skill overlap)
- [ ] Decide whether the Writer goes back to the model (proposal) or stays a template; only then make Writer examples

### Model

- [ ] Confirm Qwen3-0.6B meets accuracy, GPU speed and memory requirements on representative inputs
- [ ] Run `training/train_lora_colab.ipynb` on a Colab GPU and record the training time, token count and best validation loss
- [ ] Load the adapter in `hiremind/slm.py` (not wired in yet) and compare the fine-tuned and base model on validation
- [ ] Merge the LoRA weights if it helps CPU speed

### Agent

- [ ] Evaluate and improve the implemented roles and agent loop, including three-round retries, claim checks and memory ablations
- [ ] Integrate and test the fine-tuned adapter in the model loader

### App

- [ ] Validate the Streamlit app end to end with real resumes; show steps live and improve progress during slow CPU inference

### Evaluation and report

- [ ] Per-role evaluation on the test set (`evaluation/eval_tasks.py`)
- [ ] Ablation versions B to E
- [ ] Practical checks: CPU time, memory, model size, retries, unsupported claims in cover letters
- [ ] Evaluation report
- [ ] Final report and demo video
