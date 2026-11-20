# HireMind harness validation

The matcher uses the existing 10,013-job FAISS index. Each search considers up to 100 semantic candidates and shortlists up to 20 using overlap with résumé role titles and skills, with a small category weight. The final ranking still uses the agreed requirements, experience and preferences weights and returns at most five eligible jobs.

## Paired retrieval check

The first run completed 12 validation résumés from the existing review set. Both retrieval strategies used the same parsed profile and query for each résumé. The Searcher needed a fallback query for four résumés. The shortlist changed for all 12 résumés, including each first-five set. Both strategies returned 24 of 60 first-five jobs with the same dataset category as the résumé. The new shortlist improved by two category slots on one résumé and lost two on another.

The larger run completed 39 of the 40 validation résumés. The dense baseline matched 93 of 195 category slots (47.7%). The revised shortlist matched 100 of 195 (51.3%). Six résumés gained matching category slots, two lost slots, and 31 were unchanged. Eight searches used a fallback query and one résumé initially failed to parse. The failed case completed a separate retrieval rerun after the Parser was updated to preserve literal evidence when it cannot recover valid model output.

Category agreement is a diagnostic based on noisy labels. There were no imported, completed human judgments for this run, so job relevance precision could not be estimated. The runner records judgment coverage and precision bounds when reviewed labels are available.

## Job eligibility checks

An end-to-end baseline run placed a Certified Public Accountant posting first for the fictional staff accountant whose résumé lacks CPA evidence. The revised check placed that real posting below the requirements cutoff, even with a maximum model score. A separate real Licensed Practical Nurse posting likewise fell below the requirements cutoff for a fictional applicant with Registered Nurse evidence only. A posting naming RN and LPN as alternatives can still qualify when the résumé supports one of them. These checks use résumé evidence and cannot verify credentials.

Job parsing now recognizes common headings such as Basic Qualifications and What You'll Bring. It separates explicitly headed preferred qualifications from required sections. Unit tests cover these boundaries.

Five fictional résumés completed standard matching on Apple Silicon, and a separate in-depth pass completed for the data analyst. Each pass returned five results. The accountant's results dropped the CPA posting and the nursing results listed RN openings. A follow-up check against the real intensive-care nursery posting confirmed that five total years of nursing no longer satisfy an explicit pediatric experience duration. The same rule now separates total experience from years with specific tools or roles. An early-career pass surfaced Salesforce and Ext JS roles despite generic JavaScript evidence. The named-platform checks removed those two results on the follow-up run.

## Parser fine-tuning pilot

The pilot generated 239 weakly labelled training prompts and 24 validation prompts from the existing split. The token cap left 207 training prompts and 18 validation prompts. Qwen3-0.6B ran 40 LoRA steps on Apple Silicon and finished with category-prefix validation loss of 0.2502. The loss covered only the category prefix because the dataset lacks verified answers for the other profile fields.

The base model and the adapter each produced 10 of 10 valid profiles and matched 7 of 10 dataset categories on another validation subset. The two models made different errors. Neither used those résumés for training or checkpoint selection. This comparison supports keeping the adapter optional while the team prepares stronger labels and evaluates more roles.

## Searcher fine-tuning pilot

The Searcher pilot trained on 48 rule-driven tool actions from 24 training résumés and used 16 actions from eight validation résumés for checkpoint scoring. The 24-step run reached validation loss 0.1498. On 12 additional validation profiles, the adapter produced eight valid first searches versus six for the base model. After identical results for a rule-generated query, the adapter selected a valid stop 12 times and the base model produced no valid follow-up actions in this check. These cases exercised tool selection with rule-built profiles. They do not establish that the returned jobs are better matches.

On a paired early-career smoke test, the base Searcher and the adapter returned the same five jobs in the same order. The final shortlist excluded the Salesforce and Ext JS jobs that had appeared before the named-platform checks.

## Critic fine-tuning pilot

The label builder generated 20,819 rule-labelled training pairs and 4,456 validation pairs. A smaller subset drove the 24-step Critic pilot. Both the base model and the adapter returned valid scores on all 12 independent validation pairs. The base model's mean absolute error against the weak scores was 1.825 points out of 10. The adapter's error was 2.175 points, so the base model remains active. These are errors against training rules and do not measure human agreement.

## Reproduction

Run `python -m evaluation.retrieval` for the paired retrieval check. Run `python -m scripts.evaluate_samples --runs 1` for a full smoke run with local fictional résumés. Add `--dense-baseline` to switch only candidate retrieval while keeping the current scoring rules. Use distinct `--output` paths to keep both runs.
