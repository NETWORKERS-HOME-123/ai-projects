# Customer churn prediction baseline

## What this project does

This project trains a simple, review-friendly churn prediction baseline using synthetic but realistic account-history features. It demonstrates how to (1) prepare time-bounded features that do not include post-departure information, (2) train a calibrated logistic baseline, (3) compare that baseline to a simple recency rule, (4) evaluate precision-recall performance and calibration, and (5) produce a small staff-reviewed support queue with clear reasons and opt-outs. The goal is instructional: show reproducible, deterministic modelling and evaluation practices relevant to churn work; it does not replace production validation on real consented customer histories.

## Data

The dataset is synthetic and generated inside the code using numpy and pandas. Each row represents an account with time-aware features such as tenure, days since last activity (recency), recent usage statistics, and an explicitly synthetic churn outcome computed so the scenario is realistic for teaching. Because the data are synthetic, the README is explicit that this is a teaching example and not a replacement for real consented account histories.

## How to build it

1. Create a virtual environment with Python 3.11+ and install the requirements file.  
2. Open the project directory and inspect project.py to see how the synthetic, time-bounded features are built.  
3. Run project.py to execute a reproducible experiment: data split, calibrated logistic baseline, recency rule, evaluations and a small support queue sample.  
4. Use pytest to run the test suite which checks determinism and a conservative held-out metric floor.

## How to run

Run the trained pipeline and evaluations with:

python project.py

It prints a JSON summary of held-out metrics and a small sample of the human-review queue.

## How to test

Run the automated tests with pytest from the project directory:

pytest -q

The tests check that the run function returns expected keys, that metrics are in valid ranges, deterministic for a fixed seed, and that the main held-out metric clears a conservative floor.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| average_precision | 0.9074 |
| brier_score | 0.0612 |
| recency_rule_ap | 0.5789 |
| precision_at_recall_0.5 | 0.9672 |
| n_test | 500 |

Also returned (see `python project.py` output): `review_queue_sample` (8 items).

## Limitations

This project uses synthetic account histories created for teaching; synthetic labels are not a substitute for real consented churn outcomes and cannot establish operational performance. The recency rule and the calibrated logistic model are simple baselines intended for comparison and human review. Real deployment requires longitudinal holdout validation, fairness audits on real customer groups, and adherence to privacy and consent practices.

## Next steps

- Run the same pipeline on real, consented and time-bounded account histories.  
- Extend features to include event counts and aggregated behaviours anchored to the observation window.  
- Add group-based evaluation (by tenure, geography, or product tier) and fairness checks.  
- Build a review workflow that records reviewer decisions and outcomes for continual learning.

---

**Full case study, build steps and evaluation checks:** [Customer churn prediction model on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0043/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
