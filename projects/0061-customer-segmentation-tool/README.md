# Customer segmentation with RFM-style synthetic data

## What this project does

This small teaching project demonstrates a repeatable pipeline for customer segmentation using RFM-style numeric summaries (recency, frequency, monetary). It is deliberately simple and deterministic so students can experiment with preprocessing, clustering settings and stability checks. The pipeline scales numeric features, fits an interpretable clustering model (KMeans), compares a small grid of cluster counts, and evaluates cluster stability across two synthetic time periods.

## Data

The dataset is synthetic and built inside the code to emulate anonymised RFM summaries. Each customer row contains: recency (days since last visit), frequency (visits in period), and spend (monetary aggregated). The data is generated with clear consent-style assumptions for teaching only: no real personal data is used. This is not a replacement for real operational data and is intentionally small for clarity.

## How to build it

1. Prepare the data — the code synthesises two periods of anonymised RFM summaries with controlled cluster structure and small drift between periods. This mirrors having two reporting periods for a stability check.
2. Build the experiment — numeric features are scaled with StandardScaler and KMeans clustering is compared across a few k values. Cluster summaries (size, mean RFM) are produced for interpretability with neutral labels.
3. Evaluate the result — cluster quality on the training period is measured with silhouette score; stability across periods is measured with Adjusted Rand Index (ARI) between assigned cluster labels.
4. Define a reviewable output — the script returns numeric metrics and prints neutral aggregate segment summaries; these summaries are intended for planners and are not used for eligibility rules.

## How to run

Run the project module directly. It prints a JSON dictionary with key metrics and cluster summaries.

python project.py

(The module also exposes run(seed: int = 42) for programmatic use.)

## How to test

Install the requirements and run pytest in the project folder:

1. pip install -r requirements.txt
2. pytest -q

Tests check the returned structure, ranges and deterministic behaviour for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| best_k | 3 |
| n_segments | 450 |
| silhouette | 0.8408 |
| stability_ari | 1 |

Also returned (see `python project.py` output): `segment_summary` (3 items).

## Limitations

This teaching project uses synthetic RFM summaries, not real customer records. The clustering model is KMeans which assumes spherical clusters and numeric scaling; it may not capture complex behaviour in real commercial datasets. The pipeline and metrics are intended for exploration and planning only — segments should be reviewed by humans and validated on real business outcomes before any operational use. Also: the original starter that used handwritten digits was not appropriate for customer segmentation; this project replaces that with numeric RFM-style data.

## Next steps

Possible next steps include: adding categorical attributes and using mixed-type clustering; exploring hierarchical or density-based clustering; validating segments against business outcomes; and adding simple visual reports (outside this restricted environment).

---

**Full case study, build steps and evaluation checks:** [Customer segmentation tool on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0061/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
