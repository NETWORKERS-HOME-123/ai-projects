# Support vector machine for classification

## What this project does

This small project teaches how to train and evaluate a Support Vector Classifier (SVC) on numeric quality measurements that indicate whether an item passes inspection or should be reviewed. It shows how scaling affects SVMs, compares a linear kernel with an RBF kernel using cross-validation, and measures false negatives on a held-out test set so you can define items that need human review.

The code emphasises: a separating margin and kernel intuition, fitting a scaled SVM without leakage, and computing a simple routing rule for uncertain or risky items.

## Data

The dataset is synthetic and produced with scikit-learn's make_classification. It simulates numeric measurements (features) and a binary label: 0 = pass, 1 = defect/review. The synthetic data is small and reproducible so students can run experiments quickly; it is not real inspection data and would need replacement for production use.

## How to build it

1. Define labels — create a reproducible synthetic dataset with controlled class imbalance and clear pass/review labels using make_classification.
2. Scale and fit — construct a pipeline using StandardScaler and sklearn.svm.SVC and fit on a training split without leaking test data.
3. Compare boundaries — compare linear and RBF kernels via cross-validation on training folds and select the best kernel by mean CV score.
4. Review failures — evaluate on a held-out test set; compute accuracy, recall, precision, F1 and false negative rate, and mark low-margin (low decision function magnitude) items for human review.

## How to run

Run the project script directly with Python 3.11+. It will print a JSON dictionary of metrics. Example:

python project.py

The function run(seed=42) is the programmatic entry point and returns a dict of floats and ints.

## How to test

Run pytest in the project directory. Tests assert the presence and ranges of metrics, a conservative metric floor for the held-out accuracy, and determinism for the main metric when run with the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| kernel_chosen | rbf |
| accuracy | 0.9550 |
| precision | 1 |
| recall | 0.7097 |
| f1 | 0.8302 |
| false_negatives | 9 |
| false_negative_rate | 0.2903 |
| uncertain_count | 30 |
| train_cv_linear_mean | 0.9067 |
| train_cv_rbf_mean | 0.9433 |

> **Read the `precision` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses synthetic data to illustrate SVM training and evaluation; results do not reflect any particular manufacturing process. The routing heuristic for human review is simple (low decision function magnitude) and a realistic inspection workflow would include inspection costs, per-class misclassification costs, and representative measurements.

## Next steps

- Replace the synthetic dataset with real measurement data from your process and re-run the comparisons.  
- Explore different class-weighting strategies or calibrated probabilities for better routing.  
- Visualise 2D projections of the margin and decision function for pedagogic demonstrations.

---

**Full case study, build steps and evaluation checks:** [Support vector machine for classification on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0008/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
