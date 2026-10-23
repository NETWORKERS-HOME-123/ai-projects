# LightGBM-style size vs resource benchmark (using scikit-learn)

## What this project does
This small reproducible project benchmarks a LightGBM-style histogram boosting classifier at increasing dataset sizes and reports held-out accuracy, training time and a conservative memory estimate. LightGBM itself is not available in the permitted environment; the project therefore uses scikit-learn's HistGradientBoostingClassifier which follows a similar histogram-based approach and is suitable for teaching the same principles: controlling leaf counts, early stopping and handling a simple categorical feature.

## Data
The data are synthetic classification datasets generated with sklearn.datasets.make_classification. For each target row-count the code generates numeric features and one synthetic low-cardinality categorical feature (created by binning a numeric column). These datasets are intentionally small-to-moderate so the whole pipeline runs quickly on typical student machines.

## How to build it
1. Ensure Python 3.11+ is used and install the requirements: pip install -r requirements.txt
2. Open project.py and inspect the run(seed: int = 42) function to see dataset sizes, model configuration and measured outputs.
3. The implementation uses sklearn.model_selection.train_test_split with a fixed random_state to create a held-out validation set for each size.
4. The classifier is HistGradientBoostingClassifier (scikit-learn): leaf size is controlled via max_leaf_nodes and early stopping is enabled.

## How to run
Run the script directly: python project.py
The script prints a JSON object containing per-size metrics (train times, a conservative memory estimate in kilobytes, held-out accuracies) and a summary key largest_accuracy.

## How to test
Run pytest in the project directory: pytest -q
The test suite checks that the run() output contains expected keys, that metrics are in plausible ranges and that the main held-out metric is deterministic and above a conservative floor.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| largest_accuracy | 0.9525 |
| notes | Used scikit-learn HistGradientBoostingClassifier as LightGBM is not available in the restricted environment. Memory is a conservative estimate from numpy array nbytes. |

Also returned (see `python project.py` output): `sizes` (3 items), `train_times_s` (3 items), `memory_kb` (3 items), `accuracies` (3 items).

## Limitations
- LightGBM is not available in the permitted environment, so scikit-learn's HistGradientBoostingClassifier is used as a pedagogical substitute. Behaviour and performance will differ from LightGBM in some workloads. 
- Memory measurement is an estimate computed from numpy array byte sizes for inputs and targets; this is a coarse single-process proxy and does not include interpreter or temporary allocations. 
- Synthetic data cannot reproduce every real-world distribution; for production benchmarking use real datasets and the actual LightGBM implementation.

## Next steps
- Replace HistGradientBoostingClassifier with LightGBM's LGBMClassifier and compare results on the same datasets.
- Extend the benchmark to include varying max_leaf_nodes, learning rates and categorical encodings.
- Add multi-run averaging and simple plots (outside this constrained environment) to visualise scaling trends.

---

**Full case study, build steps and evaluation checks:** [LightGBM model for large datasets on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0074/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
