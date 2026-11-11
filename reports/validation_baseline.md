# Resume category baseline: validation

- Training split: 1,736 resumes; validation split: 372 resumes.
- Method: scikit-learn TF-IDF with default settings and logistic regression (`max_iter=1000`, seed 42).
- Accuracy: 0.608. Macro F1: 0.557.
- This is a validation result against the dataset's category labels, which are known to be noisy. It is not the held-out test result and does not measure job-matching quality.

## Full-agent smoke test

One validation resume was run through the base Qwen3-0.6B CPU agent and the real FAISS index. It returned five ranked jobs, with no Critic retries, in 1,014.2 seconds (16 minutes 54 seconds). The parsed category matched the dataset label. The Searcher stopped on a repeated query; its logging was adjusted afterward to record this as a stop when candidates exist.

The run overlapped with other local setup and testing, so the time is only a rough observation. One example cannot establish recommendation quality, Parser accuracy, or representative latency. The Critic processed 20 candidates serially and is the main observed latency bottleneck. No test-split data or human relevance labels were used.

After keeping OpenMP at one thread and allowing PyTorch four threads on Apple Silicon, a repeat on the same resume processed 20 candidates and returned five jobs in 363.7 seconds (6 minutes 4 seconds). Both runs returned five jobs and the same parsed category; the job rankings were not recorded for comparison. The runs had different background activity, so they are not a controlled benchmark. Serial Critic scoring is still too slow for a responsive demo.

## Parser check across five categories

Before adding the category constraint, the base model returned four category names outside the project's 24 supported categories and one invalid profile on five validation resumes (accounting, chef, healthcare, IT and teaching). None exactly matched the dataset label. This is a very small, deliberately diverse check and the dataset labels are noisy; it is not a test-set estimate. Reproduce with `python -m scripts.validate_parser`.

After asking the model to select from the 24 supported categories and rejecting unsupported output, all five responses had supported categories, and two exactly matched the dataset labels. The chef and healthcare examples were classified as ACCOUNTANT and the teacher example as HR. The change guards the schema but does not establish reliable classification; the model may be sensitive to category ordering. These five cases are not directly comparable to the aggregate TF-IDF baseline above.

Moving the category list after the resume changed the five predictions to FINANCE, CHEF, SALES, INFORMATION-TECHNOLOGY and TEACHER, respectively; three matched the dataset labels. This prompt was selected after observing these five validation examples, so the result is exploratory and may not generalize. Category ordering and prompt wording still affect the model's decisions. A larger, separate validation sample is needed.

On a separate set of five validation resumes from the same categories, the modified prompt returned FINANCE, CHEF, BUSINESS-DEVELOPMENT, INFORMATION-TECHNOLOGY and ARTS; two exactly matched the dataset labels. The TF-IDF baseline returned the same five categories. Disagreement is not necessarily a model error: the resume labelled HEALTHCARE describes healthcare sales leadership, and the resume labelled TEACHER begins with dance experience. This sample is also too small to estimate accuracy. Run it with `python -m scripts.validate_parser --offset 1`.

## Critic controls

On six synthetic, deliberately obvious controls (a match and a mismatch each for data analysis, nursing and cooking), the Critic returned scores on the expected side of the cutoff of 6 in all six cases. This checks basic behavior only; it is not evidence of accuracy on real or borderline job recommendations.
