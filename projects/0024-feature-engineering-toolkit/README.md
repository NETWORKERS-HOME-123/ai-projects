# Feature engineering toolkit

## What this project does

This small teaching project demonstrates how to build a simple, reviewable feature-engineering toolkit for a mixed-type tabular classification task. It emphasises practical steps: defining which raw fields are available at prediction time, composing imputers and encoders into a ColumnTransformer + Pipeline, preventing leakage from future-only fields, and comparing the engineered pipeline to a raw-feature baseline. The transformation is packaged with a version string and basic input checks so it can be reviewed for deployment.

## Data

The dataset is synthetic and created inside the code. It contains numeric features, categorical features, and a datetime field. A deliberately injected column called "future_flag" simulates a future-only signal (available during training but not at prediction) so the project can demonstrate leakage checks. The synthetic target is binary.

## How to build it

1. Create the mixed-type DataFrame in project.py using numpy and pandas, documenting which columns are allowed at prediction time.
2. Compose preprocessing: numeric imputer + standard scaler; categorical imputer + one-hot encoder (handle_unknown='ignore'); custom datetime features (day-of-week, recentness).
3. Fit a classifier pipeline (RandomForestClassifier) and a simple baseline pipeline using only basic numeric features; evaluate on a held-out test split with fixed random_state.
4. Package the transform into a small wrapper class with a version string and an input-check method; include small example fixtures demonstrating expected inputs.

## How to run

Run the module as a script. It will print a JSON object with evaluation metrics for the engineered pipeline and the baseline.

python project.py

## How to test

Run pytest in the project directory. Tests verify the returned metric keys, value ranges, a conservative performance floor on the held-out accuracy, and determinism for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| version | v2 |
| accuracy | 0.8000 |
| roc_auc | 0.8588 |
| accuracy_baseline | 0.7080 |
| n_train | 750 |
| n_test | 250 |

Also returned (see `python project.py` output): `example_input_rows` (3 items).

## Limitations

This is a small educational example using synthetic data. The synthetic patterns are simple and chosen so that the models learn reliably; real datasets will have more complex distributional shifts, require calibration, and need careful monitoring. The toolkit demonstrates common transformations but is not a full production feature-store library; it lacks persistent serialization, schema migration tooling and runtime monitoring.

## Next steps

- Add persistence (safe on-disk serialization) and schema migration checks before deployment.
- Extend the datetime transformer to support time-zone aware logic and rolling-window aggregates.
- Add more extensive unit tests covering a broader variety of missing-column and unseen-category scenarios.

---

**Full case study, build steps and evaluation checks:** [Feature engineering toolkit on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0024/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
