# HireMind checklist

Last updated: 2026-11-24

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
- [x] Added résumé evidence to candidate shortlisting and shared fallback queries while keeping the final result limit at five
- [x] Separate required and preferred sections in job parsing and check résumé evidence for clearly named credential titles
- [x] Added a paired dense-versus-shortlist retrieval comparison with judgment coverage and precision bounds
- [x] Implemented a local LoRA trainer and a reproducible category-only Parser pilot using separate training and validation splits
- [x] Trained the pilot for 40 steps and compared it against the base Parser on 10 separate validation résumés. Both matched 7 of 10 dataset categories
- [x] Generated reproducible Searcher tool-selection demonstrations from the existing index using separate training and validation résumés
- [x] Trained a 24-step Searcher adapter and checked tool actions on 12 additional validation profiles
- [x] Generated 20,819 weak Critic training pairs and 4,456 validation pairs, then trained a 24-step Critic pilot
- [x] Evaluated the Critic pilot on 12 independent pairs. The base model agreed more closely with the rule labels
- [x] Completed standard matching for five fictional résumés and checked in-depth scoring on the data analyst sample
- [x] Built a checkpointed A–E ablation runner, blind packet exporter, pooled scorer and grouped relevance calibration command
- [x] Collected the 100-résumé held-out FAISS baseline and generated blind review packets for the baseline and the earlier retrieval comparison
- [x] Evaluated the TF-IDF category baseline on all 373 held-out résumés: 61.1% accuracy and 0.537 macro F1
- [x] Built weak full-profile Parser examples and grounded Writer template examples. Trained a four-role LoRA pilot locally
- [x] Recorded a browser demo of the real upload, matching, score breakdown and editable cover-letter flow with a fictional résumé
- [x] Ran all five ablation variants on a shared three-résumé validation pilot and exported its blind review packet
- [x] Ran the component check on eight held-out résumés using dataset and rule targets, including the learned Writer's template check
- [x] Merged the experimental adapter and benchmarked an identical short CPU generation with both checkpoints
- [x] Prepared two independent blind workbooks for 272 validation pairs and added a reconciliation command that reports missing and conflicting judgments
- [x] Added a résumé-grounded cover-letter skill and verified that one résumé yields different evidence for reporting and deployment jobs
- [x] Benchmarked dense, BM25 and reciprocal rank fusion on all 372 validation résumés
- [x] Added a reproducible CPU benchmark for full matching and cover-letter drafting, with per-phase timing and peak memory
- [x] Added partial-study diagnostics, a fixed blind review batch and paired precision comparison
- [x] Audited the live Writer before and after strengthening evidence links. All six fictional regression cases pass
- [x] Compared sequential and batched CPU Critic inference using both checkpoint and float32 weights on eight fixed prompts
- [x] Completed all 500 held-out A–E cases with zero recorded errors and exported 1,607 blind review pairs
- [x] Completed the isolated CPU benchmark on three validation résumés. Each returned five jobs. Median matching time was 7m 51s and scoring accounted for 90% of the total time

- [x] Suggested labels with reasons pre-filled for all pairs in both workbooks, for review

- [x] Switched primary inference to Apple Silicon GPU for speed and retained CPU fallback. The report records GPU and CPU timings
- [x] Parser and Critic prompts moved into `build_prompt` functions shared by the roles and the training data, so training matches inference
- [x] Rule-based training data (`scripts/make_labels.py`), training and validation resumes only: 1,736 + 372 Parser examples and 20,832 + 2,232 Critic pairs (4 nearest, 4 same-category, 4 other-category jobs per resume). Uploaded to Drive `data/processed/training/`

- [x] Searcher prompt moved into `searcher.build_prompt`; rule-based Searcher traces (`scripts/make_searcher_traces.py`), regenerated on 2026-11-23 to follow the updated agent loop (shortlist search, remembered rejections, the fewer-than-five retry): 7,574 training and 1,586 validation decisions, 52% of resumes needing more than one round. Uploaded to Drive
- [x] LoRA training script (`training/train_lora.py`) and Colab notebook (`training/train_lora_colab.ipynb`): formats examples exactly like inference, masks the prompt from the loss, samples 6,000 Critic pairs, keeps the checkpoint with the lowest validation loss. Tested end to end on a tiny random model on CPU. First Colab run (2026-11-23, T4): fixed three problems found there (Colab's old `torchao` blocking `peft`; out of GPU memory at batch size 2, now batch 1 with 16-step accumulation; bf16 wrongly enabled on the T4, now fp16). Real size: 15,286 examples, 15.5M tokens per epoch, 1,912 update steps for 2 epochs, about 17 s per step (around 9 hours)

## In progress

- [x] Hassaan reviewed all 115 labels in `relevance_hassaan.xlsx` (2026-11-24): 4 changed (P045, P082, P188 to relevant; P084 to not relevant). Updated on Drive
- [x] Arslan reviewed all 115 labels in `relevance_arslan.xlsx` (2026-11-24): kept all (67 not relevant, 48 relevant). Updated on Drive
- [x] Imported all 230 reviewed labels to `evaluation/labels/relevance_val.csv`; 30 shared pairs agree 80% (Cohen's kappa 0.59)
- [ ] Tune a relevance rule using reviewed labels, including how experience relates to job level
- [ ] Review recommendation quality on the expanded validation and held-out cohorts
- [ ] Improve and independently validate Parser category accuracy; the prompt matched two of five noisy dataset labels on a separate small sample (`reports/validation_baseline.md`)
- [ ] Improve Searcher stop/retry behavior; the base model repeated a query in the real-index smoke test
- [ ] Reduce CPU scoring latency and repeat the isolated benchmark to measure the change
- [ ] Evaluate the Critic's reasons and the Writer's structured, profile-grounded drafts with human reviewers
- [ ] Calibrate the revised weighted scores and in-depth review against human-reviewed matches; the earlier fictional-resume runs predate these changes
- [ ] Review pooled results from both retrieval strategies to measure precision at five with adequate judgment coverage
- [ ] Prepare stronger labels for full profiles and validate any new adapter on a broader set before making it the default
- [ ] Measure the Searcher adapter's effect on judged final rankings as well as tool-action validity

## Decisions for the team

- [x] Hand-checked the categories of the 100 test resumes used for matching (2026-11-24): 32 dataset categories corrected (23 to another category, 9 fit none of the 24 and are marked `NONE`). Saved to `evaluation/labels/test_categories.csv` (ids only). Plan: report A–E on the 91 in-pool resumes as the main result and the 9 as a separate group; Parser category accuracy against both dataset and checked labels, excluding `NONE`
- [x] Writer: keep the grounded template rather than a trained model (decided 2026-11-24; explain the change from the proposal in the report)
- [x] Training: report both Hassaan's local pilot and the Colab run (decided 2026-11-24)
- [x] Audit missing experience levels. Only 414 of 2,684 have one clear title signal. Keep missing metadata unknown pending human validation
- [ ] Hassaan reviews `docs/category_mapping.md` and `hiremind/skills.csv` (instructions added to `hassaan_adapter_comparison.md` on Drive, step 3)

## To do

### Matching evaluation setup

- [ ] Fit and review the relevance rule on human judgments once both classes are available
- [ ] Score ablation A (FAISS only) on the test set

### Training data (rule-based labels)

- [x] Rule-based Parser labels: category, skills, years of experience, education, roles, projects
- [x] Critic labels: about 20,000 resume and job pairs with a score out of 10 and a reason
- [x] Rule-based Searcher agent and saved tool-selection training traces
- [x] Writer examples from the grounded cover-letter template
- [ ] Improve coverage of years, roles, education and projects in Parser training labels
- [ ] Review Critic training label weights against human judgments

### Model

- [ ] Confirm Qwen3-0.6B meets accuracy, CPU speed and memory requirements on representative inputs
- [x] Fine-tune a small four-role LoRA pilot and choose the checkpoint on validation using the local GPU
- [x] Colab training run (2026-11-24, T4, 1 epoch of 956 updates; written up in `reports/training_colab_2026-11-24.md`): reached update 867 (91%) in 2h 47m before the free GPU quota ran out. Validation loss fell from 0.128 (update 100) to 0.0286 (update 800, the best). The update-800 adapter is on Drive in `models/slm/qwen3-0.6b-lora/adapter-step800`; the learning rate was already near zero, so the last 89 updates would change little
- [x] Adapter comparison script (`evaluation/compare_adapter.py`): base and adapter answer the same held-out validation examples (excluding those used to pick the checkpoint). Three-example check on CPU: the adapter matched the rule labels on all three; the base model gave an unfinished Parser answer, a Critic score 3 points off and the wrong Searcher tool. About 6 minutes per example on Arslan's CPU, mostly the base model running to its token limit
- [ ] Full comparison (20 Parser, 40 Critic, 40 Searcher) on Hassaan's Mac GPU; record the results in the evaluation report
- [ ] Optional: finish the last updates with the Resume cell once the Colab GPU quota resets (about 25 minutes)
- [ ] Decide whether the merged checkpoint improves full matching enough to justify its larger artifact

### Agent

- [ ] Evaluate and improve the implemented roles and agent loop, including three-round retries, claim checks and memory ablations
- [x] Integrated opt-in adapter loading and exercised it in the held-out evaluation

### App

- [x] Fix the macOS FAISS search stall and verify the real Streamlit app on CPU with one validation résumé, including live progress, scores and the editable letter
- [x] Verify two-page text-based PDF parsing and browser upload, with recovery for unreadable, locked and textless PDFs
- [x] Validate a text-based two-column PDF with a skills sidebar and preserve the reading order of a résumé with right-aligned dates
- [x] Validate a two-page PDF with an experience column, skills sidebar, full-width sections and a wrapped project. Verify extracted evidence with the local Parser model
- [x] Record CPU latency without the GPU study running

### Evaluation and report

- [ ] Broaden the held-out component check with verified full-profile and Critic reference labels
- [ ] Obtain independent human judgments for the held-out A–E results. The exported packet has 1,607 pairs and no completed judgments
- [x] Recorded CPU time, memory, model size, retries and cover-letter regression checks in the evaluation report
- [x] Drafted the [evaluation report](reports/evaluation_2026-11-18.md) with completed run results
- [ ] Complete the final report with reviewed ranking metrics
