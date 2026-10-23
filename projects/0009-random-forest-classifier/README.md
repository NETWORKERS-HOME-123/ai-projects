# Random forest classifier

## What this project does

This small project teaches how to train a reproducible RandomForestClassifier on a built-in binary dataset, evaluate it on a held-out test set, compare it to a trivial majority-class baseline, and inspect which features the forest found most informative. The goal is to practise a safe, repeatable pipeline and to be cautious about translating results from synthetic or convenience datasets to field deployments.

## Data

The code uses the scikit-learn built-in breast cancer dataset (load_breast_cancer). This is a small, curated biomedical dataset provided for examples and experimentation. It is not real maintenance history and does not reflect production machine logs or labelled maintenance events. The dataset is used here only for teaching pipeline, evaluation and feature-inspection techniques.

## How to build it

1. Frame the target — Load load_breast_cancer, set the dataset target as the classification label and separate X (features) and y (target).
2. Train the forest — Split the data to train and test with train_test_split and fit sklearn.ensemble.RandomForestClassifier (fixed random_state). No scaling is applied because tree ensembles are scale-invariant.
3. Compare baselines — Fit a trivial sklearn.dummy.DummyClassifier that predicts the majority class and report its held-out accuracy alongside the forest's accuracy and balanced accuracy.
4. Translate cautiously — The README and code document what labelled maintenance history would be needed to attempt a production inspection pilot (example labels, sampling cadence and missingness considerations).

## How to run

Run the project file as a script. It prints the returned metrics JSON to stdout. Internally the main function is run(seed: int = 42) -> dict and returns simple, JSON-serialisable metrics.

## How to test

This repository includes pytest tests. Run pytest in the project folder; tests check that run(42) returns expected keys, metrics are within valid ranges, the primary held-out metric meets a conservative floor, and run(42) is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 0.9580 |
| balanced_accuracy | 0.9512 |
| baseline_accuracy | 0.6294 |
| precision_macro | 0.9587 |
| recall_macro | 0.9512 |
| n_features | 30 |
| note | Data: sklearn.load_breast_cancer. Model: sklearn.ensemble.RandomForestClassifier. Do not treat this as maintenance performance; collect labelled maintenance history for real pilots. |

Also returned (see `python project.py` output): `top_features` (5 items), `top_importances` (5 items).

## Limitations

This example uses a convenient, built-in biomedical dataset: it is not machine maintenance data. Feature names and distributions differ from logs or sensor streams. The model here is a demonstration: it does not include production monitoring, drift detection, or a realistic labelling strategy for maintenance outcomes. Do not infer predictive-maintenance performance from this example.

## Next steps

Possible next steps include: collect labelled maintenance events with timestamps and failure modes; engineer time-windowed features and aggregates; validate models using temporally ordered train/test splits; implement monitoring and recalibration in deployment.

---

**Full case study, build steps and evaluation checks:** [Random forest classifier on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0009/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
