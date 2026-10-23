# Association rule mining

## What this project does

This small teaching project demonstrates association rule mining on anonymised transaction baskets. It creates synthetic transaction data that mimics customers buying combinations of items across two time periods, mines frequent itemsets (up to size 3) using explicit minimum support, derives rules with support, confidence and lift, and evaluates rule stability on a later period.

The project uses a compact, transparent implementation (no external association libraries) so students can read and adapt the code and understand how support, confidence and lift are computed and interpreted. The code produces a short reviewed rule list with counts and avoids causal claims.

## Data

The data are synthetic transaction baskets created in code (not real purchases). Each transaction is encoded as item-presence columns (one-hot per item). Two time periods are created (an earlier training period and a later evaluation period) so the project can demonstrate held-out stability of rules.

## How to build it

1. Install the Python environment: pip install -r requirements.txt
2. Open project.py and inspect the run(seed=...) function which generates data, mines itemsets and rules, and evaluates stability.
3. Adjust mining thresholds (min_support, min_confidence) directly in project.py for experiments.
4. Run the file to print JSON metrics and a short reviewed rule list.

## How to run

Run the module directly with Python 3.11+: python project.py. It prints a JSON dictionary with metrics and a short reviewed rule list.

## How to test

Run pytest in the project directory: pytest -q. Tests check that the run function returns the expected keys, metrics are in valid ranges, the held-out stability meets a conservative floor, and the main metric is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_transactions_train | 840 |
| n_transactions_test | 360 |
| n_rules | 6 |
| avg_lift | 4.2392 |
| stability_fraction | 1 |

Also returned (see `python project.py` output): `top_rules` (6 items).

> **Read the `stability_fraction` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

The data are synthetic and simplified: real retail transactions may have many more items, temporal effects and customer segmentation. This code mines frequent itemsets and reports association statistics but does not make causal claims about why items co-occur. The item encoding is a simplified one-hot basket representation; the project does not attempt scalable optimisations needed for very large real-world datasets.

## Next steps

Possible extensions: experiment with larger item vocabularies, improve performance with specialised implementations (FP-growth/apriori libraries), include temporal decay or session-based windows, and build a small human-reviewed rule filter.

---

**Full case study, build steps and evaluation checks:** [Association rule mining on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0035/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
