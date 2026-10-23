# Model evaluation dashboard for classification metrics

## What this project does

This project constructs a small, reproducible evaluation pipeline for multiclass classification and produces a read-only evaluation view suitable for a dashboard. It shows per-class precision, recall and F1, confusion matrices, score curves where feasible, and compact exportable error summaries. The code also includes a tiny, hand-calculated fixture to verify behaviour when some classes receive zero predicted positives.

## Data

The project uses the built-in sklearn iris dataset (3 classes, four numeric features). This is a small, well-known dataset used here only as a deterministic example; no external data is downloaded.

## How to build it

1. Create a Python 3.11+ virtual environment.
2. Install dependencies: pip install -r requirements.txt
3. Open project.py to inspect the pipeline (train/test split, LogisticRegression classifier, metrics computation).
4. Run the script to produce a JSON evaluation object and a compact error summary.

## How to run

Run the module directly with Python. It prints a JSON object produced by run(). Example:

python project.py

The run function accepts run(seed: int = 42) -> dict so you can change the random_state.

## How to test

Run pytest in the project directory. The test suite imports run and checks keys, ranges, determinism and a conservative accuracy floor.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| dataset_version | sklearn_iris_v1 |
| accuracy | 0.9333 |
| macro_f1 | 0.9333 |
| roc_auc_ovr | 0.9956 |

Also returned (see `python project.py` output): `per_class` (object), `confusion_matrix` (3 items), `exportable_error_summary` (3 items).

> **Read the `roc_auc_ovr` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is a small, demonstration pipeline using a built-in dataset. The output is a read-only JSON-style view intended for further integration into a dashboard; it does not include a web UI. ROC/AUC computations are provided in a one-vs-rest form for the multiclass problem and require probability outputs; some score curves are less interpretable for small datasets. The hand-calculation fixture is intentionally tiny to exercise edge cases (absent predicted classes) and is not a substitute for extensive external validation.

## Next steps

- Add a simple front-end (static HTML/JS) to render the returned JSON as interactive tables and plots.
- Provide a loader for a CSV import format matching the example schema (ids, true_label, pred_label, pred_score).
- Extend evaluation to grouped or subpopulation metrics (by group id) and save exportable CSV summaries for downstream review.

---

**Full case study, build steps and evaluation checks:** [Model evaluation dashboard for classification metrics on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0040/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
