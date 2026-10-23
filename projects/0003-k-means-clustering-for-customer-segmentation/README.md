# K-means clustering for customer segmentation

## What this project does

This small teaching project demonstrates unsupervised customer segmentation using K-means clustering. Students will generate a safe, synthetic customer dataset that mimics aggregated retail signals (purchase frequency, average order value and recency), scale the numeric features, fit K-means for several choices of K, and assess clusters for stability and business interpretability rather than classification accuracy. The project emphasises practical considerations: selecting non-sensitive features, scaling, comparing cluster counts and checking whether clusters are stable and actionable.

## Data

The dataset is synthetic and generated inside the code using make_blobs from scikit-learn. It represents aggregated, permissioned customer metrics only (no identifiers, no personal sensitive traits). Features are: purchase_frequency, avg_order_value and recency_days. The synthetic data is compact and deterministic (seeded) so students can reproduce results and focus on the clustering workflow.

## How to build it

1. Generate a small, safe dataset with three numeric features that model aggregated customer metrics; use a fixed random seed for reproducibility.  
2. Select only the non-sensitive numeric features and scale them with StandardScaler.  
3. Fit K-means for several choices of K (2–6) with a fixed random_state, compute silhouette scores and choose the best K by silhouette.  
4. Inspect cluster sizes, centroids and a stability score computed from repeated K-means runs (Adjusted Rand Index between runs) and report the main metrics.

## How to run

Run the project module directly. It exposes run(seed: int = 42) -> dict which returns JSON-serialisable metrics. When executed as a script it prints the metrics as JSON.

## How to test

This repository includes pytest tests. The tests check that run(42) returns the expected keys, that numeric metrics are in valid ranges, that the silhouette score comfortably exceeds a conservative floor, and that run is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| best_k | 4 |
| silhouette | 0.7839 |
| inertia | 64.2277 |
| stability_ari | 1 |

Also returned (see `python project.py` output): `cluster_sizes` (4 items), `centroids` (4 items).

## Limitations

This is an educational, small-scale synthetic example. The generated data is a simplified stand-in for real customer data and does not capture complex real-world behaviour, seasonality or many product categories. K-means assumes spherical clusters and equal variance; if real customer segments are non-spherical or influenced by categorical variables this method may be misleading. Always avoid using any personal identifiers or sensitive attributes in segmentation.

## Next steps

Try other clustering algorithms (Gaussian Mixture, DBSCAN), include additional safe aggregated features (return rate, channel share), or evaluate business utility by mapping clusters to simple KPIs such as average monthly revenue per cluster and discussing actions with a domain team.

---

**Full case study, build steps and evaluation checks:** [K-means clustering for customer segmentation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0003/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
