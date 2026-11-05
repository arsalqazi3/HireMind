# Data notes

What we found looking through the two Kaggle datasets, and the decisions it raises.

## Resumes (LiveCareer)

- 2,484 resumes in 24 categories. About 800 words on average.
- Remove 3 before splitting: 1 with no text (ID 12632728) and 2 exact duplicates (one of 19147603 / 28398216, one of 16850314 / 37473139). That leaves 2,481.
- Employer names are replaced by the placeholder "Company Name" in nearly every resume, so the Parser cannot learn employer names from this data.
- Small categories: BPO 22, AUTOMOBILE 36, AGRICULTURE 63. A 15% test split gives only about 3 BPO and 5 AUTOMOBILE resumes.

### The category labels are noisy

The labels look keyword-based, not based on profession. The clearest signs:

- ADVOCATE is mostly patient, customer and victim advocates. Only 14% of these resumes mention attorney, lawyer or paralegal.
- ARTS includes "Language Arts" and "Culinary Arts" teachers.
- AUTOMOBILE includes an ETL developer, a tax clerk and a legal assistant.
- BPO includes a realtor and a network engineer.

Two checks on how much this matters, using all 2,481 resumes:

| Category | Category word in opening text | TF-IDF + LR F1 (5-fold) |
|---|---:|---:|
| BPO | 9% | 0.00 |
| AUTOMOBILE | 6% | 0.20 |
| CONSULTANT | 99% | 0.38 |
| ARTS | 22% | 0.41 |
| AGRICULTURE | 6% | 0.50 |
| APPAREL | 7% | 0.54 |
| SALES | 100% | 0.55 |
| ADVOCATE | 69% | 0.58 |
| HEALTHCARE | 41% | 0.60 |
| All categories | | Accuracy 0.68, macro F1 0.63 |

These are rough checks. A low "opening text" share does not prove a label is wrong, and the F1 scores are a quick test, not the proposal's final baseline, which must be measured on the test split.

**What this means:** a job search based on these labels can be wrong even when the agent does well. For example, an ETL developer labelled AUTOMOBILE would be scored against mechanic jobs.

## Job postings (LinkedIn)

- 123,849 postings, not "more than 124,000" as the proposal says. Use the real number in the report.
- 106,596 remain after dropping descriptions under 50 words and duplicate descriptions.
- `docs/category_mapping.md` maps 53,956 of these to a category and picks a pool of 10,016.

### "Skills" are not real skills

`jobs/job_skills.csv` holds only 35 broad job functions, such as "Sales", "Management" and "Information Technology". The `skills_desc` column is 98% empty.

**What this means:** `skill_overlap`, the Critic's rule score and the relevance definition all need real skills, and the dataset does not provide them. They have to come from the description text.

Suggested approach: build one fixed skill list and match it in both resumes and job descriptions, so both sides use the same words. This has not been built or tested yet.

### Experience level

`formatted_experience_level` is missing for 24.9% of postings after cleaning. Known values are Internship, Entry level, Associate, Mid-Senior level, Director and Executive.

Of the postings with no level, 56% have a level word in the title or a number of years in the description. That could fill most gaps and leave about 11% unknown. The pattern is rough and its accuracy has not been checked. A mention of years may not be an experience requirement.

## Decisions for the team

1. **Test resumes for matching.** Hand-check the category of the 100 test resumes used for matching, and replace or relabel those that are clearly wrong? Recommended, otherwise label errors count against the agent. The Parser's category accuracy should still use the dataset labels, as the proposal says.
2. **Small categories.** Keep all 24 as the proposal says, and report per-category results so BPO and AUTOMOBILE do not hide behind the average? Recommended.
3. **Skill list.** Agree on building the shared skill list described above.
4. **Job mapping.** Review the rules in `docs/category_mapping.md`, especially ADVOCATE and BPO.
