# HireMind

AI4015 Agentic Artificial Intelligence · FAST NUCES Karachi

Muhammad Hassaan (23K-0603) and Arslan Asad (23I-2505)

## Methodology

HireMind reads an English résumé, ranks up to five historical job postings, and drafts a cover letter for a posting the applicant selects. The Parser extracts a structured profile. The Searcher uses the profile to choose retrieval tools and shortlist jobs from a local FAISS index. The Critic evaluates fit, records reasons for rejected jobs, and can trigger another search. The harness limits the search to three rounds and tracks previous queries and job IDs. The Writer uses a local model for the opening and closing, inserts a relevant résumé example, and checks the result against the source. If the check fails, the app assembles a draft from the source text. The applicant can review and edit it.

The demo runs the Qwen3-0.6B base model locally. Qualification checks use job requirements and experience. User preferences contribute when the posting contains enough information to assess them. Candidates must meet qualification cutoffs before preferences can affect their ranking.

## Dataset and training pipeline

The project uses the LiveCareer résumé dataset and a historical LinkedIn postings snapshot. The evaluation split contains 1,736 training, 372 validation, and 373 test résumés. Every example derived from a résumé stays in its original split. The processed catalogue contains 10,013 postings across 24 mapped categories. Historical postings may be closed or stale.

The training scripts generate rule-labelled Parser examples, Critic judgments, and Searcher trajectories. The Colab T4 run used Qwen3-0.6B, task tokens, and answer-only LoRA training with 15,286 eligible examples per epoch. Training reached update 867 of 956. Step 800 had the lowest recorded validation loss of 0.029. The deployed Writer uses the evidence-checking workflow, a change from the proposal to fine-tune all four roles. [Training details](training_colab_2026-11-24.md)

## Evaluation

The historical study included a FAISS baseline and four agent configurations that vary Critic use, episodic memory, and checkpoint. All 500 ranking runs completed on 100 test résumés. The historical adapter arms used the earlier mixed pilot. FAISS and the full base agent each filled 500 slots. The full pilot adapter filled 394. Category agreement measures field alignment. Ranking evaluation uses pooled résumé and posting judgments and counts empty top-five slots as misses.

The separate step-800 study compared the base and Colab checkpoints on shared component prompts and a paired validation ranking cohort.

| Evaluation | Base | Step 800 |
|---|---:|---:|
| Valid Parser JSON on rule-labelled validation prompts | 7/20 | 20/20 |
| Expected Searcher tool on rule-labelled validation prompts | 15/40 | 34/40 |
| Critic MAE among valid responses | 2.831 | 0.052 |
| Validation résumés with five results | 40/40 | 33/40 |
| Precision@5 bounds on pooled validation rankings | 0.530–0.615 | 0.420–0.510 |

We generated relevance judgments with local Qwen3-8B and checked eight pairs against the source texts. After those checks, 336 of 370 pairs had determinate judgments. On the 15 résumés with complete pooled judgments, step 800 changed precision@5 by −0.067, with a paired 95% interval of −0.213 to +0.093. Step 800 improved agreement with the component rules and missed the five-result coverage criterion. The base model remains the demo default. The [evaluation report](evaluation_2026-11-18.md) records the full methods, ablations, latency and failure cases.

The held-out A–E study reviewed all 1,607 pooled pairs with local Qwen3-8B, followed by targeted source-text checks. It recorded 1,389 determinate and 218 uncertain judgments. In the 91-résumé in-catalogue cohort, precision@5 bounds were 0.631–0.741 for FAISS and 0.560–0.699 for the full base agent. Seven résumés supported a paired comparison. The full base agent's change against FAISS was +0.029, with a 95% interval of −0.199 to +0.229. The sample leaves the ranking benefit uncertain.

## Deliverables

The repository contains the Streamlit app, data preparation and index scripts, LoRA training code, tests, and evaluation scripts. Model weights, the index, and example datasets are local inputs. The [demo runbook](../docs/demo.md) describes the presentation flow and backup recording. [Repository setup](../README.md)
