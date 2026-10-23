# Price Optimisation Tool

## What this project does
This small teaching project shows how to estimate a simple demand curve and compare pricing scenarios with basic business constraints. It uses a synthetic demand dataset, fits a regression model, validates it on held-out simulated interventions, and then scores a small set of reviewable prices for revenue and margin. The aim is to explain the workflow clearly rather than to automate pricing decisions.

The model is a plain scikit-learn `LinearRegression` fit on price, promotion, inventory, and seasonality features. The output is a human-reviewable scenario summary that can be discussed by a pricing team. It does not make personalised offers, and it does not change prices automatically.

## Data
The data is synthetic and created inside the Python code. It is a faithful small stand-in for a retail demand setting, with documented assumptions for unit cost, inventory, and an acceptable price range. The simulated demand process includes a price-response effect, a promotion uplift, a limited inventory effect, and a seasonal pattern.

Because the data is generated deterministically from a fixed seed, the same run always produces the same dataset and the same evaluation split. This makes it suitable for classroom use and repeatable testing.

## How to build it
1. Prepare the data — Generate synthetic demand observations with price, promotion, inventory, cost, and seasonality assumptions.
2. Build the experiment — Fit a simple demand model and compare constrained revenue and margin scenarios across a price grid.
3. Evaluate the result — Check held-out performance on simulated interventions and test sensitivity to the learned demand curve.
4. Define a reviewable output — Return a compact scenario summary for human review, with no personalised pricing and no automatic price changes.

## How to run
Run the project script with Python 3.11 or newer:

```bash
python project.py
```

The script prints a JSON object containing the evaluation score, the learned elasticity proxy, and the suggested scenario prices.

## How to test
Install the dependencies and run pytest:

```bash
pip install -r requirements.txt
pytest -q
```

The tests check the output structure, valid metric ranges, a conservative held-out score floor, and deterministic behaviour for a fixed seed.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| held_out_r2 | 0.9171 |
| held_out_mae | 6.3203 |
| elasticity_proxy | 2.1569 |
| baseline_price | 100 |
| recommended_revenue_price | 73 |
| recommended_margin_price | 94 |
| baseline_revenue | 6307.9400 |
| best_revenue | 7286.4540 |
| best_margin | 3704.6230 |

## Limitations
This is a simulated teaching example, not a production pricing engine. The demand process is simplified, and the model assumes that the observed relationship between price and demand is stable within the generated data. It does not fully capture competitor actions, stock-outs, season-specific strategy changes, or marketing interactions beyond the included features.

The source task for this project also highlights an important lesson: a binary classifier is not suitable for price-response estimation or pricing optimisation. A demand model with an explicit objective is needed for this use case.

## Next steps
A stronger version could add hierarchical segments, richer promotion variables, cross-price effects, and uncertainty bands around the price recommendation. It could also compare several demand models, such as regularised regression or tree-based regressors, and include a manual review workflow for pricing analysts. If real business data becomes available, the synthetic demand process should be replaced with a proper experimental or observational design that is validated by domain experts.

---

**Full case study, build steps and evaluation checks:** [Price optimization tool on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0058/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
