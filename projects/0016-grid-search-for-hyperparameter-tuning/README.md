# Grid search for hyperparameter tuning

## What this project does
This small project teaches how to perform a controlled grid search over a modest hyperparameter space using scikit-learn. You will prepare a held-out test set, define training folds, run GridSearchCV on a pipeline (scaling + RandomForestClassifier), compare candidate settings by their cross-validation scores and runtimes, and evaluate the chosen model once on the final holdout. The code is deterministic: random_state is set everywhere so results can be reproduced for teaching and assessment.

## Data
This project uses the built-in scikit-learn breast cancer classification dataset (load_breast_cancer). This is a small, well-known numeric dataset suitable for binary classification experiments. Using a built-in dataset keeps the example self-contained and deterministic.

## How to build it
1. Install the Python dependencies listed in requirements.txt. Ensure you use Python 3.11+.
2. Run the project module (python project.py) to execute the pipeline and print results as JSON.
3. Run the tests with pytest to verify determinism and minimum result quality.
4. Inspect the printed experiment table to compare parameter combinations, fold scores and fit runtimes.

## How to run
Run the project file directly: python project.py
The script prints a small JSON object containing the best parameters, cross-validation score, held-out test accuracy and a ranked experiment table.

## How to test
Run the test suite with pytest: pytest -q
The tests check that the run() function returns the expected keys, that numeric metrics lie in valid ranges, that the held-out accuracy meets a conservative minimum, and that repeating run(42) is deterministic.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| best_cv_score | 0.9626 |
| test_accuracy | 0.9561 |
| total_grid_search_time_sec | 10.8184 |

Also returned (see `python project.py` output): `best_params` (object), `experiment_table` (8 items).

## Limitations
This is an instructional example with a modest grid and a single model family (RandomForestClassifier). The dataset is small and cleaned; results on larger or noisier real-world datasets will vary. Grid search over more parameters or finer values increases compute cost; more advanced tuning (random search, Bayesian optimisation) may be used in practice for larger spaces. Finally, evaluation here uses a single held-out split — repeated or nested validation would better estimate generalisation for model selection.

## Next steps
Possible extensions: expand the parameter grid, compare different estimators (e.g. logistic regression, gradient boosting), use randomized search for larger spaces, or add nested cross-validation for unbiased selection and evaluation.

---

**Full case study, build steps and evaluation checks:** [Grid search for hyperparameter tuning on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0016/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
