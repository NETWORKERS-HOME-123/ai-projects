# Fraud detection system

## What this project does

This small project demonstrates a simple fraud review pipeline that compares an Isolation Forest anomaly detector with a set of transparent rule-based review flags. It builds a synthetic but permissioned minimised transaction-like dataset, trains an Isolation Forest as a baseline, implements simple documented review rules, and evaluates both approaches by measuring precision at a fixed review capacity (the number of alerts the human team can investigate). The pipeline produces a human-review queue (no automatic blocking) with audit notes so reviewers can see why a record was prioritised.

## Data

The data are synthetic and generated inside the code using scikit-learn's make_classification. The generator creates an imbalanced binary label to mimic scarce verified fraud cases among many legitimate transactions. The features are numeric and minimised; these are a stand-in for transaction amount, frequency, account age signals and similar anonymised metrics. This is not validated transaction fraud data and should be treated as illustrative only.

## How to build it

1. Ensure Python 3.11+ and install dependencies: pip install -r requirements.txt
2. Open project.py and inspect the run(seed=42) function that builds data, trains models and evaluates metrics.
3. Run the script to get a JSON summary of metrics and a review queue example.
4. Run the tests with pytest to verify determinism and metric floors.

## How to run

Run the script directly: python project.py
It prints a JSON dictionary with evaluation metrics and counts. The main metric is precision_at_capacity which measures the fraction of verified frauds among the top N alerts that a human reviewer can inspect.

## How to test

Run pytest in the project directory. Tests check that run(42) returns the expected keys, metrics are in valid ranges, determinism holds for the main metric, and the main metric passes a conservative floor.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| precision_at_capacity | 1 |
| recall_at_capacity | 0.5714 |
| isolation_precision | 0.1250 |
| rule_precision | 0 |
| review_capacity | 8 |
| n_verified_in_queue | 8 |
| n_missed_verified | 6 |
| dataset_size | 500 |

> **Read the `precision_at_capacity` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses synthetic transaction-like data and so results do not generalise to real-world fraud operations. The features and labels are simulated; verified positive cases are artificial and scarce by design to teach evaluation methods. The Isolation Forest provides an unsupervised anomaly score; it is not a calibrated probability of fraud. The rule-based approach is intentionally simple and human-readable; it is useful for transparent triage but will not capture subtle, adversarial patterns.

## Next steps

Possible extensions include: testing with additional realistic feature engineering, adding time-series/contextual aggregation, collecting and incorporating verified labelled cases, calibrating scores for better analyst prioritisation, and measuring investigation cost trade-offs explicitly.

---

**Full case study, build steps and evaluation checks:** [Fraud detection system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0046/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
