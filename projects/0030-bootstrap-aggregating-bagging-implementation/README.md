# Bagging with decision trees (project 0030)

## What this project does

This small teaching project demonstrates bootstrap aggregating (bagging) using DecisionTreeClassifier as the base estimator. The code trains BaggingClassifier with varied numbers of trees, inspects which training samples were included in each bootstrap replicate, compares held-out (test) accuracy with the out-of-bag (OOB) estimate, and shows individual-tree predictions alongside the aggregated vote for the same held-out rows.

## Data

The dataset is synthetic and created with scikit-learn's make_classification. It is a binary classification problem with 10 numeric features and a moderate signal-to-noise ratio. The data is not from a real application; it is intentionally small and reproducible for teaching.

## How to build it

1. Prepare a reproducible synthetic classification dataset and hold out a fixed test split.
2. Train BaggingClassifier (base estimator: DecisionTreeClassifier) with bootstrap sampling and enable out-of-bag scoring.
3. Evaluate held-out accuracy and the OOB score while varying the number of estimators.
4. Collect per-tree predictions on the held-out set and compare each tree's vote to the aggregated majority vote.

## How to run

Run the script project.py. The entry point is run(seed: int = 42) which returns a JSON-serialisable dictionary of metrics and small arrays summarising per-tree behaviour. From the command line: python project.py

## How to test

Run pytest in the project directory. The test file imports run and checks: (1) the expected metric keys are returned, (2) metrics fall into valid ranges, (3) a conservative held-out accuracy floor is met, and (4) run(42) is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| holdout_accuracy_100 | 0.9000 |
| oob_score_100 | 0.8800 |
| mean_fraction_included_per_tree | 0.9975 |
| aggregated_holdout_accuracy | 0.9000 |
| mean_tree_agreement_with_aggregate | 0.8601 |

Also returned (see `python project.py` output): `per_tree_preds_first_10_rows` (100 items), `all` (object).

> **Read the `mean_fraction_included_per_tree` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is a teaching example on synthetic data intended to expose behaviour of bagging. The dataset is small and synthetic so reported numbers do not imply real-world performance. The base estimator is a full decision tree without random feature subsetting; this isolates the effect of bootstrap-sampling (bagging) from random feature selection found in random forests.

## Next steps

Possible extensions: compare to a single-tree baseline repeatedly retrained to examine variance; compare bagging with and without feature subsampling; use larger or real datasets for application-specific validation.

---

**Full case study, build steps and evaluation checks:** [Bootstrap aggregating (bagging) implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0030/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
