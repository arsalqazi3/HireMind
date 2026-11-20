# Relevance labelling

The proposal judges a job as relevant to a resume by category, skills and level, with cut-offs set on validation data. Nothing in the datasets says which jobs are truly relevant, so we label a small set by hand and tune the relevance rule's cut-offs to agree with it.

## The set

- **200 resume and job pairs** from 40 validation resumes, covering all 24 categories. No test or training resumes are used.
- **Each resume gets 5 jobs:** its 3 nearest jobs from the FAISS index, plus 2 random jobs from the same category. This mixes clear matches with borderline ones, such as the right field at the wrong level.
- **The sheets show raw text only.** They do not show the extracted skills or levels, so the labels are not steered toward the rule.
- **Split between the two of us:** each person labels 115 pairs, and 30 pairs are labelled by both to measure agreement.

## How to label

Put 1 in the `relevant` column if this person could reasonably apply for the job and be a credible candidate, and 0 if not. A job is relevant when all three hold:

1. **Field:** it is the same kind of work, or work this person's experience clearly leads to.
2. **Skills:** they have most of the main skills the job asks for, or close equivalents.
3. **Level:** the seniority roughly fits. An intern is not relevant for a director role, and a director is not relevant for an entry-level role.

Judge from the text only, and don't look at the other person's sheet. If unsure, pick your best guess and write a short note. The same guide is in the "how to label" tab of each workbook.

## Files

| File | Where | In git? |
|---|---|---|
| `relevance_arslan.xlsx`, `relevance_hassaan.xlsx` | Drive: `HireMind-shared/labels/`. Locally: `data/labels/` | No, they hold dataset text |
| `evaluation/labels/relevance_pairs.csv` | Repo | Yes, ids only |
| `evaluation/labels/relevance_val.csv` | Repo, after import | Yes, ids and labels only |

## Steps

1. Each person fills in their workbook on Drive.
2. Copy both finished workbooks into `data/labels/`.
3. Run `python -m scripts.import_relevance_labels`. It lists any empty or invalid rows, then saves the labels and prints how often the two of you agreed.

Hassaan's workbook currently contains AI-drafted suggestions based on portions of the displayed excerpts, not completed human labels. Review each pair against the full text and change the answer where needed, then remove the `AI draft` marker from that row's note. The importer rejects marked rows so provisional suggestions are not silently treated as ground truth. Arslan's workbook also contains AI-drafted labels, each with a short reason in the note and the same `AI draft` marker, and needs the same review. Both sets were drafted by AI, so the 30 shared pairs measure agreement between two AI-assisted reviews, not between two independent people. Report the labels as AI-drafted and human-reviewed. `combined.xlsx` is a point-in-time summary and does not update automatically.

To rebuild the sheets, run `python -m scripts.make_relevance_sheet`. It is seeded, so it makes the same pairs again, but don't rerun it after labelling has started.

## Review actual top-five results

The ablation runner can export a separate blind packet from saved rankings.
The packet combines each system's top five and the first matching round where
available, removes duplicates and shuffles job order. Its Markdown files
contain the source text. Keep the packet in git-ignored storage.

```bash
python -m evaluation.ablation packet outputs/retrieval_validation_all.json outputs/retrieval_review
python -m evaluation.ablation score outputs/retrieval_validation_all.json outputs/retrieval_review/labels.csv
python -m evaluation.calibrate --snapshot outputs/retrieval_validation_all.json \
  --judgments outputs/retrieval_review/labels.csv
```

The review template uses 1 for a credible match and 0 for an unsuitable one.
Leave uncertain judgments empty. The scorer gives precision bounds for
partially reviewed lists. It reports exact pooled metrics only on résumés whose
pooled jobs were all judged for every compared version. Calibration uses
reviewed validation pairs, with all pairs for a résumé kept in the same fold.
Test judgments evaluate the frozen system independently.

### Independent reviews and reconciliation

The current local handoff is in `outputs/reviewer_handoff/`. Hassaan and
Arslan each have a blank workbook with the same 272 validation pairs. Each
workbook shows excerpts. Follow the résumé ID to its Markdown file in
`outputs/retrieval_blind_review/` for the full evidence. Review independently.
For an initial batch, complete the first eight résumé groups in both
workbooks before moving on to the rest. Calibration needs at least five
résumés and five agreed pairs in each verdict class.

```bash
python -m evaluation.review reconcile outputs/retrieval_validation_all.json \
  outputs/reviewer_handoff/relevance-hassaan.xlsx \
  outputs/reviewer_handoff/relevance-arslan.xlsx outputs/review-results
python -m evaluation.calibrate --snapshot outputs/retrieval_validation_all.json \
  --judgments outputs/review-results/consensus.csv
```

Reconciliation writes a consensus CSV, a list of missing or conflicting
decisions, and ranking metrics at the current coverage. Ask the original
reviewer to complete missing votes. For conflicts, record the final human
decision in a CSV with `resume_id`, `job_id` and `relevant`, then rerun
reconciliation with `--adjudicated` and a new output directory. Blank or
conflicting decisions stay out of calibration until resolved.

### Fixed starter batches

The first eight validation groups contain 61 pooled pairs. Use the two
existing workbooks to record each independent decision. The matching
full-text sections are also collected under
`outputs/reviewer_handoff/validation_starter_eight/` for easier access.

For a first held-out batch, take the first 12 résumés in the seeded cohort.
The freeze command checks that each requested version has attempted every
case in that group, so a slow or failed case stays in the same cohort.

```bash
python -m evaluation.review freeze outputs/ablation_test_joint.json \
  outputs/ablation_test_starter.json --limit 12 --versions A B C D E
python -m evaluation.ablation packet outputs/ablation_test_starter.json \
  outputs/ablation_test_starter_review
python -m evaluation.ablation compare outputs/ablation_test_starter.json \
  outputs/ablation_test_starter_review/labels.csv --baseline A
```

For the last command, use a resolved judgment file in the packet's CSV
format. The comparison reports the top-five precision change on the
shared, fully judged résumé cohort and a paired 95% bootstrap interval.
Each resampled unit is an entire résumé. Small or incomplete batches
provide preliminary evidence. [Paired retrieval evaluation research](https://maroo.cs.umass.edu/getpdf.php?id=885)
