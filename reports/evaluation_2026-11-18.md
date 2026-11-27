# HireMind evaluation and demo

HireMind searches a fixed catalogue of 10,013 historical jobs. It returns up to five ranked jobs and can create an editable cover letter for a selected role. The ranking gives 60% weight to job requirements, 30% to experience and 10% to preferences when all three are assessable.

## Data and controls

The existing split has 1,736 training, 372 validation and 373 test résumés. No new résumé or job source was used. The held-out category comparison uses the original dataset categories, which are noisy. Training targets for the mixed pilot come from those categories, literal résumé evidence and deterministic tool and cover-letter examples. They do not substitute for independent judgments of job relevance.

| Check | Result |
|---|---|
| TF-IDF and logistic regression on all 373 test résumés | 61.1% category accuracy, 0.537 macro F1 |
| FAISS only on 100 fixed test résumés | 500 stored top-five slots. Of those, 239 share the dataset category of the résumé |
| Revised candidate shortlist on 39 paired validation résumés | 100 of 195 same-category slots versus 93 of 195 for dense retrieval |
| Mixed LoRA pilot | 64 optimizer steps, 60 usable training examples and 23 checkpoint-selection examples after token limits. Best validation completion loss was 0.4281 |
| Real browser demo | Fictional résumé upload, five results, score details and an editable letter addressed with the applicant's name |

The shortlist figures measure retrieval, before the Critic's final ranking. The pilot loss measures agreement with the generated task examples. Neither figure gives an applicant a probability of qualifying for a job.

## Relevance and ablation protocol

- **A** uses FAISS with the résumé text.
- **B** uses the mixed adapter and search without the Critic.
- **C** uses the mixed adapter and Critic without episodic memory.
- **D** uses the complete matching workflow with the base model.
- **E** uses the complete matching workflow with the mixed adapter.

Every variant uses the same seeded résumé cohort and catalogue. The agent variants share the same search implementation and maximum round budget, although retries and actual job reviews may differ. Each completed case saves the final jobs, elapsed time, retry count and first-round ranking, when applicable. For a five-hour collection window, the runner stops between cases and preserves its checkpoint.

All five variants completed a three-résumé validation pilot. Each returned 15 jobs total, with no errors. Median elapsed time per case was 0.03 s for A, 3.10 s for B, 41.00 s for C, 65.70 s for D and 37.05 s for E. C and E each retried once across the pilot. These timings describe this three-case run, including normal model warmup effects. Its pooled review packet contains 51 distinct pairs.

The blinded packets combine returned job IDs across the selected variants. The reviewer sees the original texts and a shuffled order without source, score or rank. Incomplete judgments produce coverage and precision bounds. Precision at five, reciprocal rank at five, pooled NDCG at five and retry improvement use the same fully judged résumé cohort for every version. Empty result slots count as misses. NDCG is normalized against relevant jobs found in the review pool. The catalogue outside that pool remains unjudged.

The earlier retrieval comparison produced a packet with 272 jobs across 39 résumés. The FAISS test baseline produced a packet with 500 rows across 100 résumés. Both packets are local. No completed human relevance labels were available at the time of writing. Learn the relevance rule on human-reviewed validation pairs and use grouped folds so pairs for a résumé stay together. Keep the test judgments for the frozen evaluation.

## Frozen role check

Eight held-out résumés and fixed job postings were used for the component check.

| Check | Production | Mixed pilot |
|---|---:|---:|
| Parsed résumé | 8/8 | 8/8 |
| Dataset category match | 3/8 | 4/8 |
| Valid first Searcher action | 8/8 | 8/8 |
| Valid Critic score | 7/8 | 8/8 |
| Mean absolute error on valid scores against the weak rule | 1.900/10 | 3.362/10 |
| Critic reasons mentioning a recognized skill absent from the input | 2/7 | 0/8 |
| Letters passing the grounded template line check | 8/8 | 0/8 |

The Critic error uses the valid outputs from each model. The reason check detects only recognized skill names. The letter check measures conformity with the approved template and cannot establish that other wording is false. The mixed pilot is an experimental comparison, and the app continues to use the grounded Writer and base matching model.

## CPU checkpoint check

The mixed adapter and its merged checkpoint generated identical tokens for
the same short Searcher prompt in this check. Across three warmed repeats,
median generation took 0.608 s with the adapter and 0.573 s with merged
weights. Peak process memory was 1,967 MiB and 1,670 MiB respectively.
The merged checkpoint takes about 1.1 GB on disk, compared with 4.4 MB for the
adapter alone. These figures cover one short prompt and do not predict full
matching latency.

The long held-out run writes its checkpoint to `outputs/ablation_test_joint.json`
and logs the start and completion of each case in `outputs/ablation_test_joint.log`.
On November 20 the run resumed with A and B complete and C at 86 of 100.
The previous worker had exited with signal 9. The re-run completed the next
case on the laptop GPU. The five-hour runner can stop a stalled worker at the
deadline and exports a blind packet from the saved cases when it exits.

`python -m evaluation.ablation diagnostics outputs/ablation_test_joint.json`
shows result counts, retry counts, category agreement and elapsed-time
percentiles. The paired category comparisons use the same successful résumés
from both variants. The first eight groups from the validation retrieval
study are available as a 61-pair blind review batch. A held-out starter batch
of 12 résumés can be frozen once A through E have attempted those cases.

All five variants completed the same 100 held-out résumés on November 20
with zero recorded case errors. These figures include empty slots in each
500-slot denominator.

| Variant | Slots filled | Same-category slots | Résumés with no jobs | Median per case |
|---|---:|---:|---:|---:|
| A | 500 | 239 | 0 | 0.021 s |
| B | 500 | 204 | 0 | 3.077 s |
| C | 360 | 137 | 12 | 65.511 s |
| D | 500 | 244 | 0 | 57.990 s |
| E | 394 | 154 | 8 | 60.043 s |

D returned five more same-category jobs than A on the shared 100 résumés.
C and E retried on 53 résumés each. The final packet contains 1,607 blind
review pairs. As of November 21, no human judgments were present. Precision
at five, reciprocal rank and pooled NDCG remain pending.

## Lexical and hybrid retrieval experiment

The BEIR study found BM25 a strong retrieval baseline across diverse tasks.
I used SQLite FTS5 with field weights of 3 for title, 2 for skills and 1 for
description. Reciprocal rank fusion combined 50 candidates from each source
with a rank constant of 60. [BEIR paper](https://arxiv.org/abs/2104.08663),
[SQLite FTS5](https://www.sqlite.org/fts5.html#the_bm25_function),
[reciprocal rank fusion paper](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).

Every method searched the same catalogue using the same résumé text for all
372 validation résumés. Median warm-query measurements exclude the 0.746 s
lexical index build and cover first-stage retrieval only.

| Method | Same-category jobs in 1,860 top-five slots | Median query |
|---|---:|---:|
| Dense | 983 (52.8%) | 24 ms |
| BM25 | 774 (41.6%) | 44 ms |
| Dense + BM25 | 922 (49.6%) | 68 ms |

Fusion improved the number of same-category results on 68 résumés and reduced
it on 100. A second run used only recognized skills as the query where available.
It produced 551 dense, 635 BM25 and 618 hybrid same-category results of 1,860.
This suggests skills alone lose useful role context. Both runs use dataset
category as a proxy and have zero human judgments, so precision at five is
pending. The app retains its current role-and-skill query, evidence-based
shortlist and final scoring.

```bash
python -m evaluation.hybrid_retrieval --output outputs/hybrid_retrieval_validation.json
python -m evaluation.hybrid_retrieval --query-source skills \
  --output outputs/hybrid_retrieval_validation_skills.json
```

## Missing job seniority

Of 10,013 postings, 2,684 have no labelled level. Only 414 of those missing
levels (15.4%) have a single explicit title signal such as “intern”, “junior”,
“senior” or “director”. Among 1,132 postings with a provided level and a
single title signal, 975 (86.1%) agree with that label. Another 43 postings
have conflicting title signals. The provided levels are themselves imperfect.
Keep the missing level unknown and treat title signals as separate evidence
until a reviewed sample establishes reliable rules.

Reproduce the audit with `python -m evaluation.audit_levels`.

## CPU browser reliability

A browser run on the real Streamlit app with a validation résumé stalled
after parsing. A macOS process sample located the wait at the OpenMP barrier
inside `faiss::IndexFlat::search`, on the Streamlit worker. The macOS
single-vector query now temporarily uses one FAISS thread and restores the
caller's thread setting afterward. The change leaves the index and ranking
formula intact. The rerun passed search and showed both scoring progress and
its time estimate.

The same validation résumé completed end to end in Playwright WebKit at
1440 × 900. Search began at 37.15 s, scoring at 42.33 s and the first
estimated remaining time appeared at 70.29 s. Matching took 526.31 s and
returned five results. The browser displayed the three score components and
opened an editable cover letter in 0.40 s. It reported zero page errors.
Peak sampled Streamlit process memory was 2,104.5 MiB. The held-out study was
running in the background, so this is one observed CPU case, without a
controlled speed comparison. The verification summary in
`outputs/cpu_app_validation.json` contains no résumé or letter text.

Eight sampled queries on the real index also returned identical IDs and
scores in sequential and concurrent FAISS workers.

### Full-pipeline CPU benchmark

The benchmark uses the base model and current matching and cover-letter
entry points. Its three seeded validation cases cover teaching, sales and
banking. The first case repeats the résumé from the earlier browser timing.
The script records total matching time, accumulated time spent parsing,
searching and scoring, reviews per round, retries, letter drafting time and
cumulative process peak RSS. The first case includes lazy model and index
loading. Subsequent cases reuse the same process. Results contain IDs and
measurements. This measures the application pipeline without the browser.

```bash
python -m scripts.benchmark_matching --limit 3 --output outputs/cpu_matching_validation.json
```

On November 20 the CPU run started after the held-out GPU study exited.
All three cases returned five jobs and each scored 20 candidates in one
round. Matching took 470.637 s for teaching, 455.871 s for sales and
516.521 s for banking. The median was 470.637 s. Scoring took 89.8% of the
combined matching time and process peak RSS was 2,807.5 MiB. Drafting took
1–3 ms on these three cases. The local log is `outputs/cpu_matching_validation.log`.

### CPU Critic inference pilot

On two validation résumés, the baseline and the batch each scored the same
four postings per résumé. The checkpoint's bfloat16 weights took 164.5 s
sequentially and 229.0 s in batches of four across the eight prompts. The
eight numeric scores and both ranking orders agreed. Four response texts
matched exactly. The run reversed the method order on the second résumé.
The GPU study was active, and these figures cover only Critic inference.
The pilot used the application's existing Critic prompts and ranking rules.

Process sampling showed bfloat16 matrix multiplication on the CPU. A warmed
single-thread probe of a 512 × 1,024 by 1,024 × 3,072 multiplication took
2.98 ms with float32 and 1,658 ms with bfloat16 on this machine. These are
matrix timings. End-to-end inference also pays for attention, generation
and model memory. The float32 pilot uses the same real job prompts to measure
the trade-off in speed and score agreement.

| CPU dtype | Sequential scoring | Batched scoring | Process peak RSS |
|---|---:|---:|---:|
| Checkpoint bfloat16 | 164.5 s | 229.0 s | 2,598 MiB |
| Float32 | 146.6 s | 186.5 s | 4,952 MiB |

Float32 sequential scoring took about 11% less time across these eight
prompts. One score increased from 8 to 9 and both ranking orders held.
Peak RSS is the maximum across each process's sequential and batched runs.
The comparison used the same posting IDs while the GPU study ran in the
background. The isolated full-pipeline measurement is reported above.

```bash
python -m evaluation.critic_batch --profiles 2 --batch-size 4 \
  --output outputs/critic_cpu_checkpoint.json
python -m evaluation.critic_batch --profiles 2 --batch-size 4 --dtype float32 \
  --output outputs/critic_cpu_float32.json
```

The decoder-only batch uses left padding and an attention mask. [Transformers generation guide](https://huggingface.co/docs/transformers/main/en/llm_tutorial)

The 10,013 stored posting descriptions have a median length of 473 words
and a 90th percentile of about 937. HireMind recognizes a separate requirements
section in 4,376 postings. For these, the median lengths are 523 words for the
full description and 158 for the requirements section. A shorter Critic prompt
would need to preserve duties and qualifications and handle postings with no
separate requirements heading. The current benchmark used the full prompts.

## PDF uploads

The app and command-line Parser now share the same PDF and UTF-8 TXT reader.
The two-page fictional PDF preserves the TXT fixture's evidence after
normalizing line wraps and date dashes. The real CPU Parser recovered the
name, category, stated experience, skills, education and project evidence.
The WebKit upload check replayed five previously recorded matching results
for the equivalent TXT fixture and exercised the real cover-letter builder.
It also verified that a corrupt, password-protected or textless PDF showed
its own actionable message and cleared any previous results. No page
errors were reported. Image-only PDFs require a text-based PDF or TXT copy.

An additional text-based PDF test renders a skills sidebar on the same rows
as work experience. The extractor reads the main column and then the sidebar
when the layout has a repeated gutter and an identifiable sidebar heading.
It checks that both extraction modes recovered the same words and uses the
original order for right-aligned dates or ambiguous layouts.

On November 20 a two-page fictional PDF also tested a full-width title and
project section around two columns. The local Parser recovered Alex Morgan,
Python, SQL, Microsoft Excel, both experience bullets and the complete wrapped
project. It preserved the second page and left unknown experience duration
unscored. The structured fallback produced identical profiles before and
after the section changes on the existing 100-résumé held-out cohort.

## Live Writer regression audit

Six fictional cases exercise two postings for one applicant, an unrelated
background, numbered requirements, a skills-only applicant, and missing job
details. The previous version passed four. The revised Writer passed six.
It dropped the unsupported link between floral-design experience and SQL
billing work and included the actual internship duty with the student's
listed skill and education. A pair must have shared résumé and posting
evidence before the model may select it. When a pair cannot be established,
the letter gives the applicant's background and can express interest in a
posting duty independently. The audit checks named evidence, expected and
excluded phrases, and the applicant signature. Human tone and quality review
is pending.

```bash
python -m evaluation.letter_audit --output outputs/letter_audit_recheck.json
```

## Demo and known limits

The app demo was recorded on local Streamlit at `http://127.0.0.1:8501` with Playwright WebKit at 1440 × 900. The upload, results, score breakdown and editable named draft completed with no page errors. The initial mobile view at 390 × 844 also rendered without horizontal overflow. The recording is in `outputs/demo/HireMind-demo.webm`.

Production uses the base SLM for matching. The cover-letter skill guides the
model in selecting résumé evidence for a posting and the Writer verifies the
selection before drafting. The four-role adapter is an experimental checkpoint.
Missing work-mode, salary, benefit or specialty evidence can leave parts of a
score unknown. Historical job availability, licensure and employment eligibility
require separate verification.

The earlier isolated adapter pilots and concrete matching regressions are documented in [harness validation](harness_validation_2026-11-18.md).

## November 24 evaluation update

The Colab step-800 checkpoint covers Parser, Searcher and Critic. The local
mixed pilot remains the checkpoint behind the historical B, C and E runs.
The grounded Writer uses checked résumé evidence and an editable template.
[Colab training run](training_colab_2026-11-24.md)

### Base model and step 800 on Mac GPU

The comparison ran on MPS with greedy decoding. Each checkpoint answered
the same 100 validation prompts, excluding prompts sampled for checkpoint
selection. The references came from the Parser, Critic and Searcher
training rules. The comparison used the script's per-role token limits.
Production parsing has a longer output budget and can retry.

| Measure | Base | Step 800 |
|---|---:|---:|
| Parser valid JSON (20) | 7 | 20 |
| Parser category matches (20) | 4 | 17 |
| Parser years-of-experience matches (20) | 0 | 16 |
| Parser role evidence F1 | 0.226 | 0.945 |
| Parser education evidence F1 | 0.025 | 0.833 |
| Critic valid scores (40) | 32 | 40 |
| Critic score MAE among valid responses | 2.831 | 0.052 |
| Critic scores within one point (40) | 8 | 40 |
| Searcher valid tool names (40) | 40 | 40 |
| Searcher correct tool choice (40) | 15 | 34 |
| Searcher exact tool and arguments (40) | 0 | 29 |

The adapter confused Public Relations with Digital Media on two Parser
examples. The base Searcher always called `search_jobs`. Step 800 matched
all 19 expected stops, but also stopped in all six cases that called for
`filter_jobs`. It chose the right tool with different arguments on five
search examples. The paired ranking study tests whether these decisions
affect the jobs returned and how often applicants get five results.

### Held-out catalogue coverage

The category review moved 23 résumés to a different category within the
catalogue and identified nine whose fields fall outside the catalogue.
On the 91 in-pool résumés, each variant had 455 possible top-five slots.
Matching the checked category describes the field of a returned job.
Human judgments of field, requirements and seniority are pending.
[Checked category manifest](../evaluation/labels/test_categories.csv)

| Historical variant | Slots filled, in pool | Jobs in the checked category | Five results, in pool | Slots filled, out of pool |
|---|---:|---:|---:|---:|
| A | 455 | 257 | 91 | 45 |
| B | 455 | 237 | 91 | 45 |
| C | 327 | 156 | 51 | 33 |
| D | 455 | 280 | 91 | 45 |
| E | 360 | 177 | 61 | 34 |

The full packet has 1,607 pairs. The first 12 résumés account for 198 pairs
and appear first in each reviewer workbook. At preparation, the packet has
zero human judgments and zero fully judged résumés. Missing slots limit the
best attainable in-pool precision@5 to 0.719 for C and 0.791 for E. The upper
bound is 1.000 for A, B and D. The lower bound is zero for all five variants.

An exploratory assistant review of those first 198 pooled pairs recorded 145
yes/no judgments and left 53 uncertain. All 12 résumés are in the checked
in-pool cohort. Scoring the saved judgments gives these precision@5 bounds,
with empty result slots counted as misses:

| Variant | Confirmed relevant top-five slots | Unjudged top-five slots | Precision@5 bounds |
|---|---:|---:|---:|
| A | 17 | 15 | 0.283–0.533 |
| B | 17 | 19 | 0.283–0.600 |
| C | 16 | 12 | 0.267–0.467 |
| D | 16 | 13 | 0.267–0.483 |
| E | 16 | 13 | 0.267–0.483 |

The source is `outputs/evaluation_2026-11-24/ai_review/test_starter_review.csv`.
No résumé has every pooled posting judged, so a paired precision estimate and
bootstrap interval are unavailable. These assistant judgments are exploratory;
independent human review remains pending.

The base Parser returned a profile for all 91 in-pool test résumés,
including five with an unknown category. It agreed with the original
dataset on 60 of 91 (65.9%) and with the checked categories on 60 of 91
(65.9%). The identical totals cover different cases. Eight predictions
agreed only with the original label, and eight agreed only with the
checked label.

### Exploratory in-depth review study

A separate saved study compared standard and in-depth review on all 100 test
résumés, with the step-800 adapter in both modes. Its pooled postings were
graded by an assistant or a local model against résumé and listing text.
Those grades are exploratory and do not replace the independent judgments
needed for the frozen A–E evaluation.

| Checked cohort and mode | Five results | Median time | Precision@5 bounds |
|---|---:|---:|---:|
| In pool, standard | 80/91 | 46.8 s | 0.560–0.673 |
| In pool, in depth | 77/91 | 66.4 s | 0.538–0.648 |
| Out of pool, standard | 6/9 | 77.9 s | 0.400–0.556 |
| Out of pool, in depth | 6/9 | 76.6 s | 0.444–0.578 |

For the 38 in-pool résumés whose pooled postings all received determinate
grades, standard precision@5 was 0.637 and in-depth precision@5 was 0.605.
The paired difference was −0.032, with a 95% résumé-bootstrap interval from
−0.084 to +0.005. This interval includes zero. The saved study therefore
does not establish a ranking improvement from in-depth review.
The detailed assessments and run settings are under `outputs/depth_study/`.
A targeted retrieval follow-up reused the study's five strongest and five
weakest determinate standard cases. With the step-800 adapter and depth off,
the strongest five stayed at 25 filled slots and 1.00 estimated precision@5.
The weakest five gained eight filled slots (11 to 19), and their estimated
precision@5 rose from 0.12 to 0.44. Three improved and two tied under the
saved grades. This selected ten-case check does not estimate a cohort-wide
effect. Its case details are under `outputs/retrieval_ten/`.

### Validation and checkpoint selection

The original validation sheets supplied 230 reviewed votes over 200 job
pairs. They contained suggested verdicts before review. The calibration
input has 194 resolved pairs, with 83 relevant and 111 unsuitable. Six pairs
had conflicting verdicts and were excluded. The 30 double-reviewed pairs
had 80% agreement and Cohen's kappa of 0.59.

The relevance rule used all 194 resolved pairs across 40 résumés. Five
résumé-grouped folds gave 60.8% balanced accuracy. The selected threshold
was 0.45. Skill overlap had the largest positive coefficient. A shortfall
in comparable experience reduced the fitted score, and the senior-title
conflict feature carried zero weight in this fit. The result is a validation
reference for relevance, with the production scoring weights as specified
above.

The shared handoff also requested a calibration with step 800 enabled for
the Parser. It used the same 194 pairs across 40 résumés and the same five
grouped folds. Balanced accuracy rose from 60.8% with the base Parser to
67.7% with step 800. The selected threshold rose from 0.45 to 0.55.
Skill overlap remained the strongest positive coefficient. The fitted
coefficient for an exact category match changed from +0.15 to −0.94,
which warrants review before using the rule to score matches. The
calibration measures agreement with the reviewed validation labels.

The base and step-800 workflows completed the same 40 seeded validation
résumés on MPS with zero execution errors. Runs were sequential. The
catalogue, scoring weights, search budget and Writer were fixed.

| Validation variant | Five results | Slots filled | Empty lists | Retries | Median time |
|---|---:|---:|---:|---:|---:|
| D, base model | 40/40 | 200/200 | 0 | 1 | 62.4 s |
| E, Colab step 800 | 33/40 | 184/200 | 0 | 31 | 57.8 s |

The agreed gate required at least the base model's five-result coverage and
a positive lower bound on the paired 95% precision@5 interval. Step 800
missed the coverage criterion, so the base model remains the demo
checkpoint. Each of the seven short adapter lists used both retries.
The saved rankings establish where the lists came up short. The standalone
Searcher benchmark identifies a possible source to investigate.

The 370 pooled validation pairs are ready for independent review.
Human judgments will establish whether either workflow returned more
relevant jobs within its available slots.
Only three of those 370 pairs overlap the 200 pairs in the earlier reviewed
calibration set. The overlapping verdicts cover two résumés (one relevant,
two unsuitable), so the calibration labels cannot establish ranking
precision for this cohort.

An exploratory local Qwen3-8B review judged 340 of the 370 pooled validation
pairs and left 30 uncertain. The base workflow's precision@5 bounds are
0.545–0.620, compared with 0.435–0.515 for step 800. On the 16 résumés whose
pooled postings all received determinate grades, precision@5 was 0.550 for
base and 0.488 for step 800. The paired change was −0.063, with a 95%
résumé-bootstrap interval from −0.200 to +0.100. The interval includes zero.
These are AI judgments, so they do not satisfy the independent human review
gate. The judgments and calculations are under
`outputs/evaluation_2026-11-24/ai_review/validation/`.

### Reproduction

The run's local inputs, logs, snapshots and reviewer workbooks are under
`outputs/evaluation_2026-11-24/`. The input manifest records the repository
revision, device and hashes of the catalogue, index, résumé data, checked
labels, historical snapshot and adapter weights. Keep workbooks and per-case
outputs outside Git because they can contain dataset text.

```bash
run=outputs/evaluation_2026-11-24
adapter=HireMind-shared/models/slm/qwen3-0.6b-lora/adapter-step800
export HIREMIND_DEVICE=mps HF_HUB_OFFLINE=1
python -m evaluation.compare_adapter --adapter "$adapter" \
  --data HireMind-shared/data/processed/training --out "$run/adapter_comparison.json"
python -m evaluation.calibrate --output "$run/relevance_rule.json"
HIREMIND_ADAPTER_DIR="$adapter" HIREMIND_ADAPTER_ROLES=parser \
  python -m evaluation.calibrate --output "$run/relevance_rule_step800.json"
python -m evaluation.ablation collect --split val --limit 40 --versions D E \
  --adapter "$adapter" --output "$run/validation_step800.json"
python -m scripts.validate_parser --checked-labels evaluation/labels/test_categories.csv \
  --output "$run/test_parser_base.json"
```

Reconcile each pair of reviewer workbooks against its own ranking snapshot.
Retain both original workbooks and use the consensus CSV for paired metrics.
If step 800 advances, preserve historical E and pool new step-800 jobs from
the same 100 test résumés before evaluating that checkpoint.
[Review and scoring commands](../docs/relevance_labelling.md)
