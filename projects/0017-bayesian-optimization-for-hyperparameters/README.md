# Bayesian hyperparameter optimisation vs random search

## What this project does

This small teaching project implements a Gaussian-process-based Bayesian optimiser (a surrogate model + acquisition rule) to tune two hyperparameters of a RandomForestClassifier on a synthetic binary classification task. It compares the optimiser to repeated random search under the same trial budget and records trial histories (including failed runs). The aim is to demonstrate how a surrogate-guided search can more efficiently find good hyperparameters than pure random sampling under a fixed budget, and to make the optimisation loop explicit so students see the surrogate, acquisition and evaluation steps.

## Data

This project uses a synthetic classification dataset generated with sklearn.datasets.make_classification (800 samples, 10 informative features, balanced classes). The data is intentionally small and synthetic so the code runs quickly and deterministically for teaching.

## How to build it

1. Create the synthetic dataset and a stratified K-fold splitter (fixed random seed).
2. Define bounded parameter ranges and an evaluation budget (number of trials).
3. Implement a GaussianProcessRegressor surrogate and an acquisition function (Expected Improvement). Initialise with a few random trials, then iterate: fit surrogate, find next candidate by sampling candidates and selecting the best acquisition, evaluate with cross-validation and record the trial.
4. Run a repeated-random baseline (same budget, several seeds) and collect trial histories for both methods.

## How to run

Run the module as a script. It exposes run(seed: int = 42) -> dict and prints JSON when executed. Example: python project.py

## How to test

This repository includes pytest tests. Run pytest in the project directory. The tests check that the run function returns the expected keys, that metrics are in sensible ranges, that the main held-out metric meets a conservative minimum, and that run(seed) is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| best_val_loss | 0.2971 |
| best_val_loss_random | 0.2910 |
| improvement | -0.0061 |
| budget | 12 |

Also returned (see `python project.py` output): `best_params` (object), `trials` (12 items), `random_histories` (3 items), `dataset` (object).

## Limitations

- The data is synthetic and small; results may not reflect behaviour on larger, real datasets.
- Only two hyperparameters are tuned (n_estimators and max_depth) to keep the demonstration fast and readable.
- The acquisition optimisation uses a simple sampled candidate search (not a global optimiser); this keeps the code small and deterministic.
- No visual plotting library is used in-code; instead the trial history data is provided so students can plot externally.

## Next steps

- Extend to tune more hyperparameters and use categorical choices.
- Replace the sampled acquisition maximisation with a gradient-based or multi-start optimiser for higher precision.
- Try other acquisition functions (e.g. Upper Confidence Bound) and compare performance.
- Apply the pipeline to a real dataset and consider runtime and parallel evaluations.

---

**Full case study, build steps and evaluation checks:** [Bayesian optimization for hyperparameters on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0017/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
