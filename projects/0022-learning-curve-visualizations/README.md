# Learning curve visualisations

## What this project does

This small project demonstrates how to produce numeric learning-curve results for classification tasks using scikit-learn. It computes training and validation accuracy as training sample size grows, using a preprocessing pipeline (StandardScaler) and two classifiers of different complexity: LogisticRegression (simple) and RandomForestClassifier (complex). The code reports mean scores and fold variability (standard deviation) for several training sizes and produces a concise, reviewable recommendation about whether gathering more data is likely to help.

## Data

This project uses the built-in Wine dataset from scikit-learn (multiclass, 3 classes). The dataset is small and suitable for demonstration: it keeps the example self-contained and deterministic. No external downloads are required.

## How to build it

1. Prepare the data — load the Wine dataset and compute an appropriate number of CV folds so every class has at least one sample per fold; choose training-size fractions that preserve class representation in folds.
2. Build the experiment — run sklearn.model_selection.learning_curve with a preprocessing pipeline (StandardScaler) and two estimators under the same StratifiedKFold splits to compare behaviour for small and large training sets.
3. Evaluate the result — compute mean and standard deviation of accuracy across folds for training and validation at each sample size, and compare the two models to inspect unstable small-sample results.
4. Define a reviewable output — assemble a sample-size report (numeric arrays) and produce a short recommendation about whether collecting more labelled data is likely to reduce the generalisation gap.

## How to run

Run the project module directly with Python 3.11+. It prints a JSON report to stdout. Example:

python project.py

The run(seed: int = 42) function is the programmatic entry point and returns the same report as a Python dictionary.

## How to test

Install the requirements and run pytest. Tests check that the report contains expected keys, that numeric metrics fall in valid ranges, that a conservative accuracy floor is met on the held-out validation metric, and that the main metric is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| dataset | wine |
| n_samples | 178 |
| n_features | 13 |
| n_classes | 3 |
| cv_folds | 5 |
| best_model | logistic |
| best_valid_mean | 0.9833 |
| generalisation_gap | 0.0167 |
| recommendation | Low generalisation gap; adding data is less likely to yield large gains. |

Also returned (see `python project.py` output): `train_sizes_fractions` (5 items), `models` (object).

## Limitations

This is a teaching example. The Wine dataset is small and built-in; conclusions about real applications will be limited by dataset size and domain mismatch. The code computes numeric learning curves rather than producing graphical plots, so visual inspection must be done externally if required. The recommendation is heuristic: it is based on mean accuracies and gaps observed on this dataset and should not substitute for full domain validation.

## Next steps

Suggested extensions: (1) add plotting of mean and confidence bands (matplotlib) for visual learning curves, (2) include additional preprocessing (feature selection) and regularisation sweeps, (3) repeat the experiment on a larger or domain-specific dataset to produce operational data-collection guidance.

---

**Full case study, build steps and evaluation checks:** [Learning curve visualizations on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0022/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
