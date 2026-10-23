# Multi-class classification system

## What this project does

This project demonstrates a compact, deterministic pipeline for multiclass classification using scikit-learn. It trains a baseline model on a built-in three-class dataset, preserves the mapping between original labels and probability columns, and produces evaluation outputs that help assess per-class behaviour. The code emphasises measurable outputs required for review: macro F1, per-class recall, a labelled confusion matrix, and an explicit mechanism for flagging uncertain predictions for human review.

## Data

The project uses the built-in UCI wine dataset (sklearn.datasets.load_wine). This dataset has three mutually exclusive classes and numeric features. Using a built-in dataset keeps the example small, reproducible and deterministic for teaching. The code uses a stratified holdout so each split maintains class proportions and rare classes remain represented in train and test.

## How to build it

1. Create a virtual environment with Python 3.11+.
2. Install the exact dependencies: pip install -r requirements.txt.
3. Open project.py and inspect the run(seed: int = 42) function which implements the pipeline and evaluation.
4. Run the file with python project.py to print the JSON summary to stdout.

## How to run

Run the pipeline from the command line: python project.py. The module's if __name__ == "__main__" prints a JSON object containing metrics, the label->column mapping and an example reviewable prediction.

## How to test

Run the test suite with pytest. The tests import run from project.py and assert expected keys exist, metrics fall in sensible ranges, determinism for the main metric, and a conservative metric floor is met.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| macro_f1 | 1 |
| per_class_recall_0 | 1 |
| per_class_recall_1 | 1 |
| per_class_recall_2 | 1 |

Also returned (see `python project.py` output): `confusion_matrix` (3 items), `label_order` (3 items), `label_mapping` (object), `example_review` (object).

> **Read the `macro_f1`, `per_class_recall_0`, `per_class_recall_1`, `per_class_recall_2` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is an instructional baseline using a small, clean built-in dataset; results will differ on real production data. The pipeline does not perform extensive hyperparameter search, class weighting or advanced calibration—those are natural next steps for a deployed system. The dataset is not a large-scale real-world supply; it is appropriate for teaching model behaviour but does not represent production distributional shifts.

## Next steps

1. Add class-weighting or a resampling strategy when classes are imbalanced in real data.
2. Evaluate with cross-validation and nested search for robust hyperparameter tuning.
3. Add probability calibration (e.g., isotonic or Platt) and an explicit threshold-tuning step for the review path.
4. Log predictions and review decisions to a structured store for manual review and auditing.

---

**Full case study, build steps and evaluation checks:** [Multi-class classification system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0072/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
