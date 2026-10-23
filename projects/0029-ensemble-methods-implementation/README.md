# Ensemble methods comparison

## What this project does

This teaching project demonstrates how ensembles (bagging and voting) compare with a single decision tree on a small, standard classification task. It fits three models with identical training/validation splits: a single Decision Tree, a Random Forest (bagging ensemble), and a hard VotingClassifier that combines a Decision Tree, a Logistic Regression and a K-Nearest Neighbours classifier. The code measures per-class precision/recall/f1, the held-out accuracy for each model, an error-overlap summary that shows when models make the same mistakes, and a deterministic proxy for inference cost so students can weigh accuracy gains against extra complexity.

## Data

The project uses the built-in Iris dataset from scikit-learn (3 classes, 4 numeric features). Iris is a small, well-known teaching dataset; it is not a real deployment dataset but is suitable for demonstrating ensemble behaviour, per-class metrics and error overlap analyses.

## How to build it

1. Create a Python environment with the requirements listed in requirements.txt.
2. Open project.py and inspect the run(seed: int = 42) function which prepares the data, builds the three models and evaluates them on a held-out test split.
3. Run the script to print the JSON report or import run() from project.py in tests or notebooks.
4. Run pytest to execute the supplied automated tests.

## How to run

Run the module directly with Python 3.11+: python project.py. The script prints a JSON-serialisable dictionary containing accuracies, per-class precision/recall/f1, an error-overlap summary and a simple deterministic inference-cost proxy for each model.

## How to test

Run pytest in the project directory: pytest -q. The tests check that the main metrics exist, are in valid ranges, satisfy a conservative accuracy floor on Random Forest, and that the run function is deterministic for the main reported metric when given the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy_tree | 0.9333 |
| accuracy_rf | 0.8889 |
| accuracy_voting | 0.9556 |
| n_test_samples | 45 |
| seed | 42 |

Also returned (see `python project.py` output): `per_class` (object), `error_overlap` (object), `inference_cost_proxy` (object).

## Limitations

This project uses the small Iris dataset for clarity; results on Iris do not generalise to larger or domain-specific problems. The inference cost reported is a simple deterministic proxy (component count multiplied by a unit cost) rather than a measured timing; this keeps the metric deterministic and reproducible in classroom tests. The VotingClassifier is a hard voting ensemble; other voting strategies and boosting algorithms are not explored here.

## Next steps

Suggested extensions: try different component models for voting, compare against a Gradient Boosting classifier, evaluate on larger or domain-specific datasets, measure real inference latency on target hardware, and add cross-validation to reduce variance in the comparisons.

---

**Full case study, build steps and evaluation checks:** [Ensemble methods implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0029/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
