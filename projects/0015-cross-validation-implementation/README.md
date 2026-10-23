# Cross-validation demonstration

## What this project does

This small teaching project demonstrates how to run stratified cross-validation correctly for a supervised classification problem and how to retain a held-out test for final evaluation. It emphasises: (1) preparing the data with awareness of class labels before choosing folds, (2) building a preprocessing-and-model pipeline and running cross-validation with a stratified splitter, (3) reporting per-fold scores and their spread, and (4) creating a fold report that exposes row assignments and preprocessing boundaries.

## Data

The data are synthetic and generated with scikit-learn's make_classification inside the code. This is a controlled, small dataset that imitates a multi-feature binary-class classification task for teaching purposes.

## How to build it

1. Create a virtual environment with Python 3.11+ and install the requirements listed below.  
2. Open the project files in your editor (project.py and test_project.py).  
3. Run the script project.py to see the pipeline run and JSON output printed to stdout.  
4. Run the tests with pytest to validate determinism and metric floors.

## How to run

Run the module directly: python project.py. The script prints a JSON object with cross-validation metrics, holdout evaluation and a fold report.

## How to test

Run pytest in the project directory: pytest -q. The tests check outputs are present, metrics lie in valid ranges, a conservative metric floor is met and that results are deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| cv_mean_accuracy | 0.7250 |
| cv_std_accuracy | 0.0217 |
| holdout_accuracy | 0.7688 |
| n_folds | 5 |

Also returned (see `python project.py` output): `fold_report` (5 items).

## Limitations

The example uses synthetic data produced with make_classification and a logistic regression model. Synthetic data cannot capture all complexities of real-world collections (for example, temporal dependencies, per-subject grouping or measurement shifts). The demonstration focuses on illustrating cross-validation mechanics rather than producing a deployable model. Also, the pipeline uses standard scaling and a simple classifier; other preprocessing, feature selection or model choices may be needed in practice.

## Next steps

Possible extensions include: use grouped cross-validation (GroupKFold) when data contain repeated entities, use time-aware split strategies for temporal data, evaluate alternative models and calibrations, and persist fold-level preprocessing artefacts to inspect leakage risks.

---

**Full case study, build steps and evaluation checks:** [Cross-validation implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0015/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
