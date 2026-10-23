# DBSCAN parameter explorer

## What this project does

This small project demonstrates how to use DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to discover irregular groups in two-dimensional data and to flag isolated points as noise. You will generate a synthetic 2D dataset with clusters of differing density and some isolated points, then fit DBSCAN across a small grid of epsilon and minimum-neighbour settings. The code reports cluster sizes, the fraction of noise, and computes silhouette scores when there are at least two non-noise clusters. A concise parameter explorer output helps you understand how density settings change group assignments.

## Data

The dataset is synthetic and built in-code using sklearn.datasets.make_blobs plus a set of isolated points added explicitly. This creates a small two-dimensional dataset suitable for visualising density-based clustering behaviour: some dense clusters, some sparser clusters, and clear isolated points that DBSCAN should label as noise depending on parameters.

## How to build it

1. Create a Python 3.11+ environment with the exact requirements listed in requirements.txt.
2. Save project.py and test_project.py from this package into a folder.
3. Install dependencies: pip install -r requirements.txt
4. Run the project or tests as described below.

## How to run

Run the module directly: python project.py

This will print a JSON object describing the best parameter setting found (by silhouette score), the corresponding number of clusters, the overall noise fraction and a small explorer listing results for each tested (eps, min_samples) pair.

## How to test

Run the pytest suite: pytest -q

The tests check that the run function returns the expected keys, that numeric metrics lie in reasonable ranges, that the main held-out metric meets a conservative floor, and that the function is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| median_pairwise_distance | 5.0365 |
| best_silhouette | 0.7153 |
| best_eps | 1.1000 |
| best_min_samples | 3 |
| best_n_clusters | 4 |
| best_noise_fraction | 0.0235 |

Also returned (see `python project.py` output): `explorer` (12 items).

## Limitations

This project uses synthetic data built into the script; it is not a substitute for domain-specific datasets. DBSCAN sensitivity to scale and the choice of epsilon means a single global epsilon may not work for datasets with very uneven local densities. Silhouette score is provided only when there are at least two non-noise clusters; it is not defined for a single cluster or when all points are noise.

## Next steps

- Extend the explorer to include more fine-grained epsilon values and automated suggestions (e.g. k-distance plots).
- Add small visual checks (scatter plots) when running locally to inspect cluster shapes and noise.
- Experiment with preprocessing (scaling) when features have different units or ranges.

---

**Full case study, build steps and evaluation checks:** [DBSCAN clustering implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0028/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
