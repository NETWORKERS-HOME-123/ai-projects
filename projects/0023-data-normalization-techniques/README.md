# Data normalization techniques

## What this project does

This small project demonstrates how three common numeric feature scaling techniques—StandardScaler, MinMaxScaler and RobustScaler—affect a distance-based classifier (K-Nearest Neighbours). The goal is to show that scaling is not a cosmetic step: different scalers make different assumptions about the feature distribution and outliers, and those choices change held-out model performance. The code compares the three scalers in identical pipelines, ensures no test-data leakage (all scalers are fitted on training data only) and returns both held-out metrics and concrete before/after examples for test-set outliers.

## Data

The dataset is synthetic and generated with scikit-learn's make_classification. To mimic real numeric features with different units, the code multiplies subsets of features by different constants (for example centimetres vs metres vs kilometres) and injects a small number of documented outliers. Because this is synthetic, the README is honest: the data are built-in / synthetic and only used to illustrate effects of scaling on a distance-based classifier.

## How to build it

1. Create a virtual environment with Python 3.11+ and install the requirements: pip install -r requirements.txt.
2. Open project.py and inspect run(seed: int = 42) where data generation, train/test split and scaler pipelines are implemented.
3. Run the script directly (python project.py) to see JSON output of metrics and saved training transformation snippets.
4. Use pytest to run the included tests (pytest -q).

## How to run

Run the script directly: python project.py. The script prints a JSON object containing per-scaler held-out accuracy, the chosen best scaler by accuracy, a small set of test-row outliers with their before/after transformed values (using the training-only fit), and saved scaler parameters (lists) so a reviewer can inspect the training transformation.

## How to test

Run pytest in the project directory. The tests check that the run function returns the expected keys, that metrics are within reasonable bounds, that the primary held-out metric meets a conservative floor, and that running run(42) twice returns identical results.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy_standard | 0.9040 |
| accuracy_minmax | 0.8960 |
| accuracy_robust | 0.9440 |
| best_scaler | robust |
| n_train | 375 |
| n_test | 125 |

Also returned (see `python project.py` output): `outlier_test_indices` (1 items), `outlier_examples` (object), `saved_transform` (object).

## Limitations

This project uses synthetic data to illustrate scaling behaviour; real-world datasets may have far more complex distributions and feature correlations. K-Nearest Neighbours is deliberately chosen because distance calculations make scaling effects easy to observe; other estimators (for example decision tree ensembles) may be less sensitive to scaling. The synthetic outliers are intentionally simple to make the transformations easy to inspect.

## Next steps

Possible next steps include: repeat the comparison on a real dataset with mixed units, evaluate additional scalers (e.g. PowerTransformer), measure robustness under cross validation over multiple random seeds, and inspect per-feature distributions with visual plots (not included here to keep dependencies minimal).

---

**Full case study, build steps and evaluation checks:** [Data normalization techniques on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0023/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
