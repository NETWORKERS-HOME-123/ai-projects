# AdaBoost staged analysis with label-noise experiment

## What this project does
This small teaching project demonstrates AdaBoost (ensemble boosting) using shallow decision-tree weak learners (stumps). It fits AdaBoostClassifier on a synthetic binary classification problem, inspects staged predictions across boosting rounds, and measures sensitivity to controlled label noise. The example emphasises how AdaBoost reweights examples and how that affects held-out error and the weights assigned to weak learners.

## Data
The data are synthetic and created with scikit-learn's make_classification. I build a moderately separable binary problem (800 rows, 10 numeric features). A controlled label-noise experiment is created by flipping a fixed fraction of training labels to simulate noisy annotation.

## How to build it
1. Prepare the data — generate a synthetic binary dataset and make a stratified train/test split. Create a noisy version of the training labels by flipping a fixed percentage.
2. Build the experiment — train AdaBoostClassifier using DecisionTreeClassifier(max_depth=1) as weak learners. Train separate models on clean and noisy training labels, using the same random seed.
3. Evaluate the result — compute per-round held-out accuracy using staged predictions to inspect how error evolves with boosting rounds; extract estimator weights (alpha) from the fitted AdaBoost models.
4. Define a reviewable output — the run() function returns a JSON-serialisable dictionary containing final accuracies, best held-out accuracies and the boosting round where they were achieved, plus the weak-learner weights.

## How to run
Run the module as a script: python project.py. The script prints a JSON object with the experiment metrics. The run(seed: int = 42) function can be imported and called from other code or tests.

## How to test
This project includes pytest tests (test_project.py). Tests verify the expected keys, sensible numeric ranges, a conservative metric floor on held-out accuracy for the clean training case, and determinism (run(seed) produces the same output on repeated calls).

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| final_accuracy_clean | 0.8550 |
| best_accuracy_clean | 0.8650 |
| best_round_clean | 17 |
| final_accuracy_noisy | 0.7950 |
| best_accuracy_noisy | 0.8100 |
| best_round_noisy | 43 |
| noise_rate | 0.2000 |
| n_estimators | 50 |

Also returned (see `python project.py` output): `weak_weights` (50 items).

## Limitations
This uses synthetic data which is useful for teaching but not a substitute for a real domain dataset. The label-noise model is a simplistic simulation (random flips) and does not capture structured or adversarial noise. The experiment uses a single dataset split (with reproducible random seed); additional cross-validation would give a more robust assessment.

## Next steps
- Extend to cross-validation or repeated splits to report mean and variance of metrics.
- Compare to bagging and random forests to highlight differences in how ensembles handle noisy labels.
- Visualise staged error curves and individual estimator decision boundaries for deeper intuition.

---

**Full case study, build steps and evaluation checks:** [AdaBoost algorithm implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0031/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
