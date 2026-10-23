# Market basket analysis tool

## What this project does

This project demonstrates a small, deterministic market-basket analysis pipeline using synthetic anonymised transaction data. It shows how to mine co-purchase item pairs, compute support, confidence and lift, validate those rules on a later time window and compare promoted and non-promoted weeks. The output is a merchandising review report (human readable metrics and rule summaries) rather than an automated pricing or bundling action.

## Data

The dataset is synthetic but designed to resemble anonymised supermarket baskets: each transaction has a date, a small set of item identifiers and a flag for returned transactions. Promotions are simulated on selected weeks to boost sales of a few items. The code generates the data deterministically from a random seed so experiments are reproducible.

## How to build it

1. Create a virtual environment and install dependencies.
2. Place project.py and test_project.py in the same folder.
3. Run the project to produce the review report (it prints JSON to stdout).
4. Run the tests with pytest to validate behaviour and metrics.

## How to run

Run the project file directly with Python 3.11+:

python project.py

It prints a JSON object with counts, average lift and a held-out agreement metric.

## How to test

Run pytest in the project directory:

pytest -q

Tests check that the pipeline returns expected keys, metrics are in sensible ranges, a conservative metric floor is met and that results are deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_transactions | 1934 |
| n_items | 39 |
| n_rules | 190 |
| avg_train_lift | 1.0356 |
| promo_support | 1 |
| nonpromo_support | 0 |
| held_out_agreement | 1 |

## Limitations

The dataset is synthetic and simplified: it does not capture complex customer baskets, multi-store effects or long-term customer loyalty. The mining implementation focuses on item-pair rules (size-2 itemsets) for clarity and speed; larger itemsets would require a more complete frequent-itemset algorithm. The output is a review report to help merchandisers, not an automated recommendation service.

## Next steps

Possible extensions include: mining larger itemsets, adding time-aware decay for old rules, grouping items by category, and integrating real transaction logs with anonymisation and privacy controls.

---

**Full case study, build steps and evaluation checks:** [Market basket analysis tool on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0049/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
