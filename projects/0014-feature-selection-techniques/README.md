# Feature selection: filters vs model-based

## What this project does

This project compares two feature-selection approaches for a supervised classification problem: a statistical filter (SelectKBest with ANOVA F-test) and a model-based wrapper (recursive feature elimination, RFE). Selection is kept strictly inside training folds during cross-validation to avoid leaking test information. The project reports held-out accuracy on a reserved test set, the feature-count reduction each method achieves, and the selection stability (how consistent the chosen features are across CV folds). It also produces a small feature audit that shows how removing groups of features (informative, redundant, irrelevant) affects held-out performance.

## Data

The data are synthetic and generated with sklearn.datasets.make_classification. The generator is configured to create informative, redundant and irrelevant (noise) features so we can exercise and measure selection behaviour. The code assigns names and group labels to columns so the audit can report the effect of removing each group.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Open project.py — it contains a run(seed: int = 42) -> dict function which executes the experiment.
4. Run python project.py to print the JSON results to stdout.

## How to run

Run the experiment with a fixed seed (default 42):

python project.py

The script prints a JSON object summarising metrics and the feature audit.

## How to test

Run the pytest suite:

pytest -q

The tests check that the run() output contains expected keys, that metrics lie in sensible ranges, that the main held-out metric passes a conservative floor, and that the run is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| selectk_mean_cv | 0.7967 |
| rfe_mean_cv | 0.8067 |
| selectk_test_accuracy | 0.8100 |
| rfe_test_accuracy | 0.8050 |
| selectk_feature_count | 7 |
| rfe_feature_count | 7 |
| selectk_stability | 1 |
| rfe_stability | 0.9000 |

Also returned (see `python project.py` output): `audit` (object).

## Limitations

This is a small educational experiment using synthetic data. Real-world datasets may have different noise structures, correlations and feature types (categorical, text, images). The synthetic generator is helpful for illustrating selection behaviour but cannot capture all practical complexities. The methods shown (SelectKBest and RFE with a linear estimator) are examples; different scorers, classifiers or hyperparameter tuning would be appropriate for other problems.

## Next steps

Possible extensions include: try different scorers (mutual information), use different estimators for RFE (tree ensembles), add nested CV for hyperparameter tuning, and apply the same pipeline to a real dataset with domain-specific feature groups.

---

**Full case study, build steps and evaluation checks:** [Feature selection techniques on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0014/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
