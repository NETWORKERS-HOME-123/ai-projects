# Linear regression for house prices

## What this project does
This beginner project teaches a simple linear regression pipeline for predicting numeric house sale prices from a small set of property features. You will build a synthetic but realistic toy dataset (floor area, location, age, property type), fit a scikit-learn LinearRegression baseline, and evaluate it honestly on a held-out test set using MAE and RMSE. The aim is to illustrate the difference between a fitted line and a reliable out-of-sample price estimate, and to show how a median baseline compares to the trained model.

## Data
The dataset is synthetic and generated inside the code using numpy. It is a modest-sized simulated housing dataset where price is a linear function of floor area with added effects for location, age and property type plus noise. This represents a classroom stand-in for real market data; it is not real transaction data.

## How to build it
1. Install the required packages (see requirements). Create a virtual environment if you prefer. Example: python -m venv venv && source venv/bin/activate (or venv\Scripts\activate on Windows).
2. Install dependencies: pip install -r requirements.txt
3. Run the project module: python project.py to print evaluation metrics for the held-out test set.
4. Run tests: pytest -q to run the automated checks and see the pipeline validated.

## How to run
Run the main script directly: python project.py. It prints a JSON object with the main evaluation metrics and simple diagnostics.

## How to test
Run: pytest -q. The test suite checks that the run function returns the expected keys, that metrics lie in reasonable ranges, that a conservative performance floor is met, and that results are deterministic for the same seed.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| test_mae | 7470.5067 |
| test_rmse | 9425.9956 |
| test_r2 | 0.9991 |
| baseline_mae | 253735.2591 |
| n_train | 900 |
| n_test | 300 |

> **Read the `test_r2` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations
This is a small synthetic dataset intended for teaching. Real housing markets have many more relevant predictors (detailed location, condition, supply/demand, transaction timing, policy effects) and more complex, non-linear relationships. A simple linear model on synthetic data is useful for pedagogy but not for production pricing decisions.

## Next steps
Possible extensions include: adding more realistic features (number of rooms, bathrooms, proximity to transport), using regularisation (Ridge/Lasso), exploring non-linear models (tree-based methods), and validating on real transaction data with proper geographic splits and temporal holdouts.

---

**Full case study, build steps and evaluation checks:** [Linear regression model for predicting house prices on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0001/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
