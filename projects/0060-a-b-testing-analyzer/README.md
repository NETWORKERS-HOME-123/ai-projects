# A/B testing analyzer

## What this project does

This project builds a small, deterministic A/B testing analysis pipeline in pure Python using numpy, pandas and scikit-learn for data manipulation. It simulates a fixed-horizon, randomised experiment with one row per unit and a binary outcome (conversion). The analyser performs data validation checks (sample-ratio mismatch and missing outcomes), estimates absolute and relative treatment effects, computes normal-approximation confidence intervals, runs a prespecified hypothesis test (two-sided z-test for proportions) and estimates the false-positive rate via A/A simulations. The final output is a concise decision report that records effect estimates, uncertainty and the original stopping rule.

This is intended as a teaching project for students to learn how to validate and report simple A/B experiments. It is not a production analysis library and does not attempt advanced adjustments such as sequential monitoring corrections, covariate adjustment using regression, or bootstrap-based inference.

## Data

The data are synthetic and generated inside the code. Each row represents a single randomised unit assigned to control or treatment. The binary outcome (converted or not) is generated using Bernoulli draws with a baseline conversion probability and an optional additive treatment effect. A small proportion of outcomes are set to missing to exercise the missing-data checks. The simulation is reproducible via a seed parameter.

## How to build it

1. Create a virtual environment and activate it (optional but recommended).
2. Install the required packages: pip install -r requirements.txt
3. Open project.py to inspect the simulation parameters and the run(seed=...) entry point.
4. Run the script with Python 3.11+ to produce a JSON decision report.

## How to run

Run the project directly: python project.py

This prints a JSON document with keys for absolute and relative effects, confidence interval bounds, p-values, data quality checks and the A/A false-positive rate estimate.

## How to test

Run pytest in the project directory: pytest -q

There are automated tests that check the return structure, reasonable metric ranges, a conservative metric floor on the A/A false-positive estimate, and determinism for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_total | 2000 |
| n_A | 987 |
| n_B | 1013 |
| missing_rate | 0.0180 |
| srm_pvalue | 0.5610 |
| abs_effect | 0.0542 |
| rel_effect | 0.6021 |
| ci_lower | 0.0259 |
| ci_upper | 0.0825 |
| p_value | 0.0002 |
| alpha | 0.0500 |
| stopping_rule | Fixed-horizon: n=2000, alpha=0.05 |
| aa_false_positive_rate | 0.0550 |

## Limitations

This teaching pipeline uses normal approximations for confidence intervals and hypothesis tests which are reasonable for moderate to large sample sizes but can be inaccurate for very small samples or extremely imbalanced conversion rates. The simulator is simplistic: it models a single binary outcome and does not represent time-to-event, repeated measures, interference between units, or heterogenous treatment effects. The analysis does not correct for multiple testing or adaptive stopping rules; the stopping rule in the report is just recorded, not enforced as a sequential-correction.

## Next steps

Possible extensions include adding covariate adjustment (logistic regression or stratification), exact/binomial tests for small samples, bootstrap confidence intervals, sequential monitoring corrections (alpha spending), and richer outcome types (counts, continuous metrics). Students can also adapt the pipeline to ingest real experiment logs and implement pre-registered analysis plans.

---

**Full case study, build steps and evaluation checks:** [A/B testing analyzer on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0060/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
