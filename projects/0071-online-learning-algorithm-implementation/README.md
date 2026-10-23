# Online incremental learning with prequential evaluation

## What this project does

This project demonstrates online (incremental) learning for a binary classification stream using a synthetic dataset with a controlled distribution change (drift). It implements a prequential (predict-then-update) evaluation: for each incoming batch the models predict first, metrics are recorded, and then an incremental model is updated via partial_fit. A frozen baseline model, trained once on the initial data window and never updated, provides a reference to observe adaptation to drift. The incremental estimator used is SGDClassifier (logistic loss).

## Data

The data is synthetic and created inside the code. It is a two-phase binary classification stream with numeric features: the first phase follows one linear decision boundary, and after a change-point the class-conditional means shift to produce a distribution drift. This controlled setup is small and deterministic so experiments are reproducible.

## How to build it

1. Create a Python environment with the requirements listed in requirements.txt.
2. Install packages: pip install -r requirements.txt
3. Open project.py to inspect the streaming loop and the prequential evaluation implementation.
4. Run project.py (python project.py) to print the results JSON.

## How to run

Run python project.py. The script prints a JSON object with evaluation metrics from the prequential run.

## How to test

Run pytest in the project directory (pytest -q). The test suite imports run from project.py and verifies returned keys, metric ranges, determinism and a conservative performance floor on the held-out prequential accuracy.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| online_mean_accuracy | 0.9610 |
| frozen_mean_accuracy | 0.9470 |
| drift_fraction | 0 |
| resets | 0 |
| n_batches | 50 |

## Limitations

This is a synthetic and small-scale demonstration: real-world streams may have higher-dimensional features, label noise, class imbalance and more complex drift patterns. The reset control implemented here is simplistic (reinitialise the incremental model on sustained performance decline) and would need careful design for production. The model uses SGDClassifier with a linear model; non-linear phenomena may require different incremental estimators or feature engineering.

Note also that the baseline in this example is a single frozen model trained on the initial window; in practice you may prefer more sophisticated ensembles, windowing policies or explicit drift detectors.

## Next steps

- Replace the synthetic drift with real streamed data or richer simulated scenarios.
- Try different incremental estimators, e.g. sparse or kernel approximations, and tune learning rates.
- Implement more robust drift detectors and adaptive windowing strategies.
- Add logging and checkpointing to make resets and model versions auditable.

---

**Full case study, build steps and evaluation checks:** [Online learning algorithm implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0071/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
