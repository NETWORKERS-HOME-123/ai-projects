# Product recommendation engine

## What this project does

This project builds a small product recommendation pipeline that ranks candidate catalogue items per user. It starts from permissioned user-item interactions (synthetic in this project), splits each user's interactions chronologically to create a held-out test event, builds a user-similarity based recommender and adds three practical rules: popularity fallback for sparse users, excluding already seen items, and filtering items that are out of stock. The pipeline evaluates ranking performance with recall@5, reports catalogue coverage, and measures behaviour for new (cold-start) users against a popularity-only baseline. A minimal opt-out recommendation widget is implemented as a pure-Python stateful object with dismiss and reset controls for demonstration and easy unit testing.

## Data

The data here is synthetic and generated inside the project code using numpy and pandas. It mimics permissioned interaction logs: user id, item id, rating (binary purchase/interaction), and a per-event timestamp. Item catalogue entries include an availability flag (in-stock/out-of-stock). The synthetic data intentionally models sparsity and popularity skew so the recommendation heuristics and evaluation are meaningful without external datasets.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. Install dependencies with pip install -r requirements.txt.
3. Open project.py to inspect the data generation, model and evaluation logic.
4. Run project.py to produce JSON metrics on the held-out test events.

## How to run

Run the pipeline directly: python project.py

It prints a JSON object containing metrics: recall_at_5, coverage, popularity_recall_at_5 and new_user_recall. The run(seed=...) function is deterministic when the same seed is provided.

## How to test

Run pytest in the project directory. Tests verify the expected output keys, value ranges, a conservative metric floor on held-out recall@5, and determinism for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| recall_at_5 | 1 |
| coverage | 1 |
| popularity_recall_at_5 | 0.0408 |
| new_user_recall | 0 |

> **Read the `recall_at_5` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is an educational, small-scale pipeline using synthetic interactions. Real-world deployment requires richer signals (timestamps at finer granularity, session context, richer item metadata), stronger evaluation (more users, business metrics, A/B tests), and production considerations (caching, incremental updates, privacy controls). The simple binary interaction assumes an implicit positive signal; we do not model explicit negative feedback. The popularity fallback helps cold users but can amplify popularity bias in practice.

## Next steps

Possible next steps include adding content-based features for items, hybridising with matrix factorisation, time-aware ranking that discounts old interactions, richer cold-start treatments (side information for new users/items), and a small web demonstration to visualise the opt-out widget behaviour.

---

**Full case study, build steps and evaluation checks:** [Product recommendation engine on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0047/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
