# Fictional resume qualification — 17 November 2026

> Historical snapshot: these runs predate the later weighted-scoring and frontend changes and do not validate the current scoring implementation.

## Method

Five fictional resumes were each run through the local agent five times: 25 actual matching runs on Apple MPS. Each run used the existing job index and local base model. Matching time excluded cover-letter assembly. The run records, profile data and generated drafts are in the git-ignored `outputs/sample_evaluation.json`. After the matching batch, a writer-only wording change was applied and all 25 letters were regenerated from the saved profiles and top jobs in `outputs/sample_letter_recheck.json`; matching was not rerun after that wording-only change.

All 25 matching runs completed without an exception and returned five jobs. Per resume, the five runs returned the same job IDs in the same order. Matching time was 50.5–111.3 seconds, median 60.5 seconds. This is a small local run, not a controlled benchmark or a statistical measure of relevance. Greedy model generation makes repeated results unsurprising.

## Manual observations

| Fictional resume | First result | Assessment |
|---|---|---|
| Avery Reed, accountant | Staff Accountant (9.8/10) | Direct accounting overlap, but the posting asks for five years while the profile states four; 9.8 overstates confidence. |
| Samira Khan, data analyst | Data Scientist (8.5/10) | Related analytics work, but the SQL/Power BI developer at rank five more directly matches the resume's experience and stated three years. |
| Taylor Ahmed, early career | Web Designer (7.5/10) | The old senior AI and cloud architect results were removed, but this posting asks for Wix Studio experience not shown in the resume. Several other results require unlisted technologies. |
| Jordan Lee, registered nurse | Travel Nurse RN (8.6/10) | Nursing is relevant, but US state licensing, clinical specialty, certifications, location and travel preferences are not verified. Two results are LPN roles rather than RN roles. |
| Alex Morgan, software engineer | Software Engineer (8.4/10) | The first result asks for five years and many technologies. The third result describes a junior Python/JavaScript role with a two-year requirement and is arguably a closer match. |

The source pool contains historical 2023–24 postings. Current availability, location, work authorization and regulated credentials were not verified. Scores are heuristic and uncalibrated; they are not probabilities or evidence of eligibility. Role/category corrections and experience extraction can fail or be inaccurate, especially when resumes are sparse or headings differ.

## Cover letters

The writer now selects up to two literal work-history bullets, prioritizing skills mentioned in the selected posting, alongside supported profile facts and projects from an explicitly labelled project section. It does not generate new achievements. The applicant still needs to verify the extracted profile and edit placeholders and wording. The checker only detects deviations from its deterministic template; it is not a general factual verification system. Resume headings outside the recognized section format may be missed.

Manual review of the five first-run letters confirmed that the selected experience statements originated in the fictional resumes. The letters remain formulaic and may repeat resume phrasing. The nursing letter also repeats extra date and benefit text from a noisy job title. A short resume does not support a detailed invented narrative.

The structure follows university career-writing guidance: a job-specific opening, specific experience linked to the job, and a closing. See [Purdue OWL](https://owl.purdue.edu/owl/job_search_writing/job_search_letters/cover_letters_1_quick_tips/quick_content_tips.html) and the [University of Michigan Career Center](https://careercenter.umich.edu/content/cover-letter-resources).

## Qualification verdict

The workflow is functional for these five fictional inputs, with consistent repeated outputs and better use of resume history. It is not validated for reliable job recommendations: labelled relevance evaluation, broader resume formats, occupation and experience constraints, skill alternatives, preference capture, and regulated credentials remain open. Human review is needed before using recommendations or letters.
