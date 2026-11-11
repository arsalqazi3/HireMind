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

To rebuild the sheets, run `python -m scripts.make_relevance_sheet`. It is seeded, so it makes the same pairs again, but don't rerun it after labelling has started.
