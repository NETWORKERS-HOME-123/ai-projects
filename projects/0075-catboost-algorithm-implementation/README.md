# CatBoost comparison (teaching project)

## What this project does

This small teaching project shows how to train and compare a model that emulates CatBoost-style handling of categorical features with a conventional one-hot baseline. We create a mixed dataset (numeric + categorical), introduce missing values and unseen categories in the test split, then train two approaches: a one-hot-encoded Logistic Regression baseline, and a gradient-boosting classifier using ordinal encoding to emulate ordered categorical handling. The aim is to demonstrate practical differences in handling high-cardinality categories, missing values and unseen categories at test time.

## Data

The data is synthetic and built inside the code. Numeric features are sampled from a classification generator and we add several categorical columns with varying cardinalities, deliberate missing entries and an unseen category injected into the held-out test set. This is a small, reproducible dataset intended for didactic experiments rather than production use.

## How to build it

1. Ensure Python 3.11+ and the requirements are installed (see requirements field supplied with the project).  
2. Read the project.py file to inspect the dataset construction, preprocessing and modelling pipelines.  
3. The project trains two models: a one-hot baseline (LogisticRegression) and a gradient-boosting model (HistGradientBoostingClassifier) with ordinal encoding for categories.  
4. The code produces a JSON-serialisable report containing a documented categorical schema and held-out metrics.

## How to run

Run the project directly: python project.py

The script runs deterministically with seed=42 by default and prints a JSON report to stdout.

## How to test

This repository includes pytest tests. Run: pytest -q

The tests check that the output contains the expected keys, that metrics are sensible, that a conservative metric floor is met and that runs with the same seed are deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| dataset_rows | 1200 |

Also returned (see `python project.py` output): `categorical_schema` (object), `onehot_baseline` (object), `ordinal_gb` (object).

## Limitations

This project does not use the CatBoost library itself. Instead it uses scikit-learn's HistGradientBoostingClassifier as a reproducible stand-in to demonstrate the concepts of ordered categorical handling. The synthetic data is necessarily simpler than many real-world problems: it is small and lacks domain-specific noise or complex feature interactions. The pipeline also does not persist models or include a full deployment workflow.

## Next steps

- Replace the stand-in model with CatBoostClassifier for a real comparison when the CatBoost package is available.  
- Experiment with more realistic categorical encodings (target encoding, leave-one-out) and cross-validation schemes.  
- Add calibration plots and per-category error analysis to examine the behaviour on rare/unseen categories.

---

**Full case study, build steps and evaluation checks:** [CatBoost algorithm implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0075/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
