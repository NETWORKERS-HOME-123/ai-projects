# Gradient boosting for customer churn

## What this project does

This small teaching project demonstrates how to build and evaluate a gradient-boosting classifier to predict customer churn from permissioned subscription-style data. It contrasts a GradientBoostingClassifier with a simple baseline, emphasises holding out a test set (no test-set tuning), assesses precision and recall, and inspects probability calibration using Brier score. The code also sketches how a staff-reviewed opt-in intervention could be measured (conceptually) without promising operational outcomes.

## Data

The dataset is synthetic but shaped like consented subscription data (one record per account). It includes tenure, usage, plan tier and support contacts. A simple, transparent rule with noise defines churn within a 90-day window. This keeps the example self-contained and safe for teaching; in production you must use real, permissioned data and time-based labelling.

## How to build it

1. Create a virtual environment with Python 3.11+ and install dependencies.
2. Save project.py and test_project.py from this repository into a folder.
3. Run pip install -r requirements.txt to install numpy, pandas and scikit-learn (and pytest for testing).
4. Run the project with python project.py to see metrics printed as JSON.

## How to run

Run the script with Python 3.11+:

python project.py

This prints a JSON object of held-out metrics for the baseline and the gradient-boosting model.

## How to test

Run the test suite (pytest is required):

pytest -q

There are tests that check the returned metrics, a conservative metric floor on the held-out accuracy, and determinism for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| baseline_mostfreq_accuracy | 0.7640 |
| logreg_accuracy | 0.8000 |
| gbc_best_params | {'learning_rate': 0.05, 'n_estimators': 50} |
| gbc_accuracy | 0.7960 |
| gbc_precision | 0.6176 |
| gbc_recall | 0.3559 |
| gbc_brier_score | 0.1285 |
| invite_fraction | 0.1000 |
| invite_precision | 0.6000 |

## Limitations

The data are synthetic and simplified; they do not capture real customer behaviour, seasonal effects or the full privacy and consent considerations of real retention modelling. Probability calibration is assessed with Brier score only; in production you should plot calibration curves and consider time-to-event analyses rather than immediate binary churn labels.

## Next steps

1. Replace the synthetic data with real, permissioned, time-ordered subscription records and define churn with an explicit holdout window.  
2. Add calibration plots, reliability diagrams and isotonic or Platt calibration if needed.  
3. Design an A/B test and a staff-reviewed opt-in workflow, then measure the actual effect on retention rather than model scores.

---

**Full case study, build steps and evaluation checks:** [Gradient boosting algorithm implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0010/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
