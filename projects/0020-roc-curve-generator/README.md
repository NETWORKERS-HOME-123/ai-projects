# ROC curve generator

## What this project does

This small teaching project demonstrates how to compute and inspect receiver operating characteristic (ROC) curves and area under the ROC curve (AUC) for a binary classifier. The code trains a logistic regression on synthetic, imbalanced data, computes continuous scores for the positive class, constructs ROC and precision–recall (PR) curves, and produces a compact threshold table that shows false alarms and missed cases at selected thresholds. It also includes small checks for reversed labels and tied scores to illustrate how these situations affect ranking metrics.

## Data

This project uses synthetic binary classification data generated with scikit-learn's make_classification. The dataset is deliberately imbalanced (more negatives than positives) to make precision–recall behaviour informative. The positive class is labelled 1. No external data is required.

## How to build it

1. Create a fresh Python 3.11+ virtual environment and activate it.
2. Install the dependencies with: pip install -r requirements.txt
3. Open project.py and read the run(seed=...) function; it is deterministic with random_state everywhere set to the provided seed.
4. Run the script with python project.py to see JSON output printed to the console.

## How to run

Run the module directly: python project.py

The script prints a JSON object containing numeric metrics and a compact threshold table for review.

## How to test

Run the tests with pytest: pytest -q

The tests validate presence of expected keys, metric ranges, a conservative AUC floor, and determinism of the main metric for a fixed seed.

## Results

<!-- RESULTS -->

## Limitations

This is a teaching example using synthetic data. Performance numbers do not represent any real application. The model is a simple LogisticRegression used to produce continuous scores; no calibration, feature engineering, hyperparameter search, or external validation is performed. The ROC and PR computations are computed on a single held-out split; cross-validation or repeated experiments would be needed for more reliable estimates.

## Next steps

- Replace synthetic data with a real, relevant dataset and ensure the positive class definition matches the application.
- Add calibration and reliability diagrams if well-calibrated probabilities are important for deployment.
- Extend threshold selection methods (cost-sensitive selection, expected utility) and show their effect on false alarms vs missed detections.
- Run repeated splits or cross-validation to produce confidence intervals on AUC and PR metrics.

---

**Full case study, build steps and evaluation checks:** [ROC curve generator on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0020/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
