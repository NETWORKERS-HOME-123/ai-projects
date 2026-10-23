# Apriori: frequent itemset mining (small teaching project)

## What this project does

This project implements the Apriori algorithm from first principles on a tiny, hand-crafted basket dataset. The goal is pedagogical: you will see candidate generation (joins), subset pruning using the Apriori property, and support counting. The code compares the Apriori output with an exhaustive enumeration (ground truth) and records how the number of candidates grows as the support threshold is reduced. The implementation is deliberately small, deterministic and easy to read so final-year students can trace each step.

## Data

The dataset is synthetic and built-in to the project code: a compact collection of seven transactions (baskets) with manually chosen items (labels like 'A', 'B', 'C', ...). The small size makes item frequencies hand-countable and makes exhaustive enumeration practical. This is a faithful small stand-in for real market-basket data intended for learning the algorithmic mechanics.

## How to build it

1. Install the project dependencies with pip (see requirements.txt).  
2. Open project.py and read the implementation — the Apriori loop is clearly separated from the support counting and exhaustive-check helper.  
3. Run the script to view JSON output summarising iteration candidates, pruned sets and support counts.  
4. Inspect tests in test_project.py to see how determinism and correctness against exhaustive enumeration are asserted.

## How to run

Run the module directly with Python 3.11+:

python project.py

The script prints a JSON object describing the run: agreement with exhaustive enumeration, the support threshold used, the number of frequent itemsets discovered and a small diagnostic about candidate growth.

## How to test

Run the test-suite with pytest (installed via requirements.txt):

pytest -q

Tests check that the Apriori implementation matches exhaustive enumeration, that outputs are in sensible ranges, that a conservative metric floor is met and that runs are deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| agreement | 1 |
| num_transactions | 7 |
| min_support_used | 0.4000 |
| num_frequent_itemsets | 5 |
| max_candidate_count | 6 |

Also returned (see `python project.py` output): `iterations` (3 items).

## Limitations

This project uses a toy dataset of seven baskets to keep the computation and reasoning simple. Real market-basket datasets are much larger and require efficient data structures and optimisations (e.g. transaction reduction, compressed representations) that are not implemented here. The implementation is single-threaded and memory-inefficient for large item universes.

## Next steps

- Extend the code to read larger datasets (CSV) and to use bitsets or boolean matrices for faster counting.  
- Add mining of closed or maximal frequent itemsets.  
- Implement variations such as association rule generation with confidence/lift thresholds.

---

**Full case study, build steps and evaluation checks:** [Apriori algorithm implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0036/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
