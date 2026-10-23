# A/B testing framework (project 0033)

## What this project does

This teaching project implements a small, auditable A/B testing framework for fixed-horizon experiments. It focuses on stable assignment to treatment arms, exposure logging, and a simple difference-in-means analysis. The project also runs repeated A/A simulations (true no-effect experiments) to check two important properties: sample-ratio mismatch detection and confidence-interval coverage for the difference-in-means estimator.

The goal is to illustrate why predictive classification is not the same as estimating a causal effect from a randomised intervention: here we construct a randomised allocation, log exposures deterministically from unit identifiers, and analyse the causal contrast using standard statistical estimators.

## Data

The data are synthetic and generated inside the code using numpy and pandas. Each experimental unit has a stable integer id, a few numeric covariates, a baseline outcome probability and a binary observed outcome generated within a prespecified outcome window. We perform A/A simulations by setting the true treatment effect to zero so the coverage properties of confidence intervals can be evaluated.

## How to build it

1. Install the Python dependencies listed in requirements.
2. Open the project Python file (project.py) and inspect the run(seed: int = 42) function — it contains data generation, stable assignment, logging and analysis.
3. Run the module directly (python project.py) to print a JSON manifest and summary metrics to stdout.
4. Run the tests with pytest to verify determinism, metric floors and basic validity checks.

## How to run

Run the example pipeline from the command line:

python project.py

This prints a JSON object with an experiment manifest, results summary and evaluation metrics.

## How to test

Run the automated tests with pytest in the project directory:

pytest -q

The test suite checks that run(42) returns the expected keys, that metrics lie in reasonable ranges, that the main metric meets a conservative floor, and that calling run with the same seed returns the same output deterministically.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| ci_coverage | 0.9050 |
| sample_ratio_mismatch_rate | 1 |
| mean_diff | 0.0171 |
| sd_diff | 0.0220 |

Also returned (see `python project.py` output): `manifest` (object), `audit_table_subset` (5 items).

> **Read the `sample_ratio_mismatch_rate` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses synthetic, simulated data and a simple normal-approximation confidence interval for the difference-in-means. It does not perform covariate adjustment, stratified randomisation beyond stable hashing, or sequential/hypothesis-adjusted stopping. The normal approximation can be inaccurate for very small samples or highly imbalanced arms. The code focuses on clarity and auditability rather than production-ready instrumentation.

## Next steps

Possible extensions include: adding covariate-adjusted estimators (ANCOVA), implementing sequential stopping rules with type-I error control, richer exposure and compliance models, and integrating a logging backend for long-running experiments.

---

**Full case study, build steps and evaluation checks:** [A/B testing framework on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0033/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
