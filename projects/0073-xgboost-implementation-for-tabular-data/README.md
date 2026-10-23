# XGBoost-style boosted trees for tabular classification

## What this project does

This small project demonstrates training a boosted-tree classifier on tabular data and comparing it to a simple forest baseline. The goal is pedagogical: to show how to prepare a labelled table (including deliberate missing values), hold out a final test set, train a boosted-tree model with early stopping on a training-derived validation split, and report discrimination, calibration and runtime. Because the xgboost Python package is not available in the constrained environment for this assignment, this project uses scikit-learn's HistGradientBoostingClassifier as a faithful, deterministic stand-in for XGBoost-style gradient boosting; the README and code explicitly note this.

## Data

The dataset is synthetic and created with sklearn.datasets.make_classification. It contains 1000 rows and 12 numeric features, of which several are informative. To emulate real-world data issues, a documented 5% of values are set to missing at random. The dataset is split stratified into a training set (80%) and an untouched test set (20%); the training set further uses an internal validation fraction for early stopping.

## How to build it

1. Create a reproducible synthetic binary classification table with 12 features and 5% missing values, then make a stratified train/test split and keep the test set untouched.  
2. Prepare a simple preprocessing pipeline using mean imputation for missing values.  
3. Train a HistGradientBoostingClassifier (used here as a stand-in for XGBoost) with early stopping via a training-derived validation fraction, and also train a RandomForestClassifier baseline on the same training data. Record runtime and model parameters.  
4. Evaluate the held-out test set using ROC AUC (discrimination), Brier score (calibration proxy), and accuracy; produce a small JSON-like report with dependency versions and feature definitions.

## How to run

Run the project file with Python 3.11+: python project.py. The script prints a JSON object with metrics, runtime and a short report. The run(seed: int = 42) function can be imported and called programmatically.

## How to test

Run pytest in the project directory: pytest -q. The test suite checks that run(42) returns the expected keys, that metric values lie in sensible ranges, that a conservative metric floor is met for the held-out ROC AUC, and that repeated calls with the same seed are deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_samples | 1000 |
| n_features | 12 |
| auc_test | 0.9365 |
| brier_test | 0.0987 |
| accuracy_test | 0.8600 |
| auc_baseline | 0.9258 |
| brier_baseline | 0.1089 |
| accuracy_baseline | 0.8600 |
| time_seconds_boosted | 0.8322 |
| time_seconds_baseline | 0.7985 |

Also returned (see `python project.py` output): `feature_names` (12 items), `packages` (object), `report` (object).

## Limitations

- This project uses synthetic data (make_classification) rather than real-world tabular data; results therefore illustrate behaviour but are not claims about a particular applied dataset.  
- The environment for this assignment does not include the xgboost package; scikit-learn's HistGradientBoostingClassifier is used as a deterministic substitute. While algorithmically similar (both are gradient-boosted trees), differences in implementation and hyperparameters mean results will not exactly match xgboost.

## Next steps

- Replace the synthetic dataset with a real tabular dataset suitable for the problem and re-run experiments.  
- Add a small hyperparameter search (randomised) for both boosted and forest models, keeping the tuning budget identical.  
- Persist model artefacts and calibration curves and add cross-validation to better estimate generalisation.

---

**Full case study, build steps and evaluation checks:** [XGBoost implementation for tabular data on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0073/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
