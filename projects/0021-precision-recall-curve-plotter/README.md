# Precision–Recall Curve Explorer

## What this project does
This beginner project demonstrates how to compute precision–recall information for a binary classifier and how to interpret average precision in a low-prevalence setting. It creates a synthetic binary dataset with a small positive class prevalence, fits a simple probabilistic classifier, computes the precision–recall curve and the average precision score, compares that score to the trapezoidal area under the precision–recall curve, and produces review-queue statistics for candidate thresholds.

## Data
The data are synthetic and generated inside the code using scikit-learn's make_classification. The generation intentionally produces a low positive-class prevalence so you can see how prevalence affects precision and recall. This is a small, reproducible dataset for teaching purposes only.

## How to build it
1. Create a Python 3.11+ virtual environment and activate it.
2. Install the project requirements: pip install -r requirements.txt
3. Open project.py to inspect the code; it uses sklearn, numpy and pandas only.
4. Run the script directly or run tests with pytest to check behaviour.

## How to run
Run the module: python project.py. The script prints a JSON object with metrics and helper tables for thresholds and review-queue estimates.

## How to test
Run pytest in the repository root. Tests check that the main metrics are present, in valid ranges, deterministic for a fixed seed, and that average precision exceeds a conservative floor.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| average_precision | 0.1792 |
| pr_auc_trap | 0.1696 |
| prevalence | 0.0533 |
| seed | 42 |

Also returned (see `python project.py` output): `precision_recall_points` (object), `thresholds_info` (3 items), `review_queue` (3 items).

## Limitations
This project uses synthetic data to illustrate concepts rather than to provide a deployable model. The classifier is simple (logistic regression) and is used to produce calibrated-ish scores for demonstration. The average precision reported depends on the synthetic generation parameters; results will vary if you change the seed or the data generation settings. The code computes score-based precision–recall curves but does not produce visual plots because plotting libraries are not required for the exercises here.

## Next steps
Possible extensions include: adding real-world datasets with documented class prevalences, calibrating classifier scores (Platt scaling or isotonic regression), comparing average precision across multiple datasets with prevalence normalisation, and producing visual plots for teaching presentations.

---

**Full case study, build steps and evaluation checks:** [Precision-recall curve plotter on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0021/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
