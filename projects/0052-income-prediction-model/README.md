# Income prediction with an interpretable baseline

## What this project does

This teaching project demonstrates how to build a simple, reproducible income prediction pipeline using a small synthetic regression dataset derived from scikit-learn generators. The exercise treats the target as an approximate continuous income amount (a simulated annual income in thousands) and fits an interpretable linear baseline (ordinary least squares). It also produces subgroup error reports for a sensitive attribute (a synthetic gender proxy) and discusses limitations and prohibited uses. The objective is pedagogical: to show modelling steps, fairness-oriented subgroup checks and a short classroom analysis report.

## Data

The code uses a synthetic regression dataset produced with sklearn.datasets.make_regression and then augments it with a small engineered sensitive attribute (gender proxy: 0/1). The labels are synthetic continuous values intended to stand in for income amounts (for teaching only). This is not a real economic dataset and must not be used for decisions about people.

## How to build it

1. Install the Python environment (see requirements). 2. Run the script project.py to create the model and evaluation. 3. Inspect subgroup RMSE and R2 reported for the held-out test split. 4. Review the classroom analysis notes in the printed report for provenance and prohibited-use guidance.

## How to run

Run the module as a script. It prints a small JSON report of held-out performance and subgroup errors.

## How to test

Run pytest in the project directory. Tests check deterministic behaviour, expected output keys, sensible metric ranges and a conservative metric floor for the held-out R2.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_samples | 2000 |
| n_train | 1400 |
| n_test | 600 |
| r2 | 0.9934 |
| rmse | 1.2369 |
| coef_mean_abs | 4.7601 |
| rmse_gender_0 | 1.2907 |
| rmse_gender_1 | 1.1844 |
| analysis_note | Synthetic regression dataset (make_regression). Target is a simulated
continuous income amount in thousands. Model: LinearRegression pipeline.
Do not use for decisions about people. See README for provenance and limits. |

> **Read the `r2` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses synthetic data (built with make_regression) and a synthetic sensitive attribute. The labels are not observed incomes from real households; they are simulated amounts for classroom experimentation. Any patterns or subgroup gaps observed here reflect the synthetic construction and should not be interpreted as real-world conclusions. The linear model is intentionally simple and interpretable but will not capture complex non-linear relationships that may be present in real economic data. Do not deploy or use the model for eligibility or any decision affecting people.

## Next steps

Possible classroom extensions: use a public real dataset with income labels, add multiple sensitive attributes and perform causal or fairness-aware analyses, compare interpretable baselines with regularised linear models and non-linear learners, and evaluate temporal stability using longitudinal data if available.

---

**Full case study, build steps and evaluation checks:** [Income prediction model on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0052/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
