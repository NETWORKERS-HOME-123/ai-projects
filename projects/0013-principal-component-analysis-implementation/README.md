# PCA exploration on the Wine dataset

## What this project does

This small teaching project shows how to fit Principal Component Analysis (PCA) as a preprocessing step, measure how much variance is retained by a reduced representation, and compare downstream model performance and reconstruction error against a baseline without dimensionality reduction. The goal is to let students see why retaining high variance is useful in some cases, and why it may not always be sufficient for a downstream task.

## Data

This project uses the built-in scikit-learn wine dataset (numeric chemical measurements of wines). The data are therefore built-in and not downloaded. We use a supervised classification task (three classes) to compare a baseline classifier against the same classifier trained on PCA-reduced features.

## How to build it

1. Install the Python dependencies listed in requirements.txt into a Python 3.11+ environment.
2. Run the project script to produce metrics: python project.py
3. Inspect the code in project.py to see the pipeline: train/test split, scaler fitted on training only, PCA fitted on scaled training data, and a LogisticRegression classifier evaluated on held-out data.
4. Run the tests with pytest to verify determinism and minimum expected performance.

## How to run

Run the project directly:

python project.py

This prints a JSON object with metrics including baseline accuracy, PCA accuracy for 5 components, reconstruction mean squared error for 5 components, and the cumulative explained variance for 5 components.

## How to test

Run the automated tests with pytest in the project directory:

pytest -q

The tests check that the run function returns the expected keys, that numeric metrics are in plausible ranges, that a conservative performance floor is met on the held-out accuracy for PCA with 5 components, and that the main metric is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_features | 13 |
| baseline_accuracy | 1 |
| accuracy_pca_5 | 0.9778 |
| reconstruction_mse_5 | 0.2291 |
| explained_variance_cumsum_5 | 0.8016 |

Also returned (see `python project.py` output): `components_tried` (10 items), `accuracy_by_k` (object), `reconstruction_mse_by_k` (object), `explained_variance_cumsum_by_k` (object).

> **Read the `baseline_accuracy` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses a built-in, small scientific dataset (wine). Results may differ on other datasets or larger-scale problems. PCA is an unsupervised rotation that orders directions by variance; high variance directions do not always contain the information a downstream task needs. LogisticRegression is used here as a simple, interpretable classifier to show differences in held-out accuracy; other models or cross-validation strategies could be used for more robust evaluation.

## Next steps

- Try other datasets (digits, iris, or synthetic generated sets) to see different variance structures.
- Evaluate using cross-validation and multiple random seeds to quantify variability.
- Compare PCA to supervised dimensionality reduction techniques (e.g., LDA) or feature selection methods.
- Visualise loadings and reconstructions with plots (this code prints numbers so students can create plots locally).

---

**Full case study, build steps and evaluation checks:** [Principal component analysis implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0013/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
