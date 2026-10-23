# Logistic regression for binary classification

## What this project does

This small project teaches a simple, transparent binary classification pipeline using Logistic Regression. You will generate a reproducible toy dataset, split it into training and held-out test sets, fit a pipeline that scales inputs and trains a logistic model, and then inspect how different probability thresholds change precision, recall and the confusion matrix. The aim is to show why probabilities (not only class labels) matter, and how to choose a threshold depending on whether false positives or false negatives are more costly.

## Data

The dataset is synthetic and produced with scikit-learn's make_classification. It simulates a consented, harmless enquiry outcome (a binary label) and 10 numeric features. This keeps the example small, reproducible and free of sensitive personal data.

## How to build it

1. Create a synthetic binary dataset with make_classification (800 rows, controlled class balance and informative features).
2. Split the data into train and held-out test sets using a stratified split and a fixed random seed.
3. Fit a StandardScaler and a scikit-learn LogisticRegression on training rows only; keep random_state everywhere for determinism.
4. Evaluate held-out probabilities and compare precision, recall and confusion matrices for a few thresholds.

## How to run

Run the project file. It exposes a run(seed: int = 42) -> dict function; when executed as __main__ it prints the result as JSON.

## How to test

Use pytest. The repository contains tests that check the returned keys, that metric values are in sensible ranges, that accuracy comfortably exceeds a conservative floor, and that runs with the same seed are deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_train | 640 |
| n_test | 160 |
| threshold | 0.5000 |
| accuracy | 0.9250 |
| precision | 0.9333 |
| recall | 0.8750 |
| f1 | 0.9032 |

Also returned (see `python project.py` output): `confusion` (object), `threshold_comparison` (object).

## Limitations

This is a teaching example with synthetic data. Real deployments need careful data governance, bias audits and clinical or legal review where relevant. Logistic Regression is a linear model; if relationships are highly non-linear more complex models may be needed. Also, threshold tuning here is illustrative; in practice choose thresholds based on costs, stakeholder input and validation on realistic held-out data.

## Next steps

- Add calibration checks (reliability diagrams) and Platt scaling if probabilities are miscalibrated.
- Extend to cross-validation for more stable estimates.
- Add cost-based threshold selection or use precision-recall curves when classes are imbalanced.

---

**Full case study, build steps and evaluation checks:** [Logistic regression for binary classification on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0002/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
