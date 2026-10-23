# Hierarchical clustering explorer

## What this project does
This small teaching project demonstrates agglomerative (hierarchical) clustering on a built-in numeric dataset. You will compare three linkage rules (average, complete and Ward), inspect silhouette scores for a chosen flat cut, check cluster stability under resampling, and produce a minimal dendrogram explorer that reports the linkage tree and representative items for a selectable cut height (implemented as choosing a number of clusters). The goal is to learn how linkage choice and scaling affect hierarchical structure and cluster quality.

## Data
This project uses the scikit-learn built-in Iris dataset (numeric descriptors for iris flowers). The data are small, standard and purely numeric — a realistic toy for teaching hierarchical clustering.

## How to build it
1. Create a virtual environment with Python 3.11+ and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Open project.py to inspect the code; it loads and scales the Iris features.
4. Run the code (see "How to run") to get JSON metrics and a minimal dendrogram structure.

## How to run
Run the main script directly with Python:

python project.py

It prints a JSON object of metrics and small structures for exploration. The run(seed) function returns the same dict and is deterministic for a fixed seed.

## How to test
Use pytest in the project directory:

pytest -q

This runs deterministic unit tests that check expected output keys, valid metric ranges, a conservative metric floor and determinism for the main metric.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| silhouette_average | 0.5381 |
| silhouette_complete | 0.5046 |
| silhouette_ward | 0.5111 |
| stability_ward_mean_ari | 0.8530 |
| n_samples | 150 |

Also returned (see `python project.py` output): `dendrogram_linkage_ward` (149 items), `representative_indices_ward` (object).

## Limitations
This is a small educational pipeline on a classic toy dataset. It is unsupervised: cluster labels do not correspond directly to class labels, and choice of number of clusters or cut height affects interpretation. The dendrogram explorer provided here is minimal (lists of merges and representative item indices) rather than an interactive GUI. The Iris dataset is small and cannot demonstrate all complexities of large, high-dimensional, or noisy real-world data.

## Next steps
- Add an interactive dendrogram visualiser (web-based or notebook) to allow selecting cuts by height.
- Extend stability checks (e.g. subsampling without replacement, bootstrap with feature perturbation).
- Apply to larger real datasets and experiment with linkage + distance metric combinations (e.g. correlation-based distances for time series).

---

**Full case study, build steps and evaluation checks:** [Hierarchical clustering implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0027/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
