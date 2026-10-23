# KNN neighbourhoods — a compact teaching project

## What this project does

This small project teaches the K-nearest neighbours (KNN) idea with a hands-on comparison: you will build a tiny numeric dataset that represents item descriptors, implement a simple neighbour search from scratch, and compare that implementation to scikit-learn's KNeighborsClassifier. The exercise emphasises why feature scaling matters, how distance choice affects matches, and how to select k using only training folds. Finally, it inspects misleading nearest neighbours for a few held-out examples.

## Data

The data is synthetic and created with scikit-learn's make_blobs to simulate numeric item descriptors and discrete classes (a toy product-similarity scenario). This is intentionally small and structured so results are easy to interpret; it is not real product data.

## How to build it

1. Create numeric item descriptors and labels using make_blobs with a fixed random seed so results are reproducible. Split data into train and test sets.  
2. Fit a StandardScaler on the training features only, then transform both train and test splits so scaling information does not leak. Implement a simple Euclidean neighbour search (no optimisation) to predict labels from the k closest training points.  
3. On the training set only, tune k by cross-validation (stratified folds) choosing from a small odd-valued grid (e.g. 1,3,5,7,9). Fit scikit-learn's KNeighborsClassifier with the chosen k.  
4. Evaluate the final classifier on the held-out test set, compare with the hand-built neighbour predictor, and list how many test points have a nearest neighbour that disagrees with the true label (a simple measure of potentially misleading matches).

## How to run

Run the project script directly with Python 3.11+. The script exposes a run(seed: int = 42) function. If executed as a program it will print JSON with the reported metrics.

## How to test

Automated tests (pytest) import run from the project and verify the output keys, sensible metric ranges, a conservative accuracy floor on the test split, and determinism when run with the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 1 |
| best_k | 1 |
| cv_best_score | 1 |
| hand_knn_accuracy | 1 |
| misleading_nearest_count | 0 |
| n_test | 90 |

> **Read the `accuracy`, `cv_best_score`, `hand_knn_accuracy` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is a toy example: the dataset is small and synthetic to keep the code short and deterministic. The hand-built neighbour search is O(n_test * n_train) and not suitable for large datasets. Only Euclidean distance is demonstrated; other metrics (cosine, Manhattan) could be useful in different descriptor spaces.

## Next steps

Try alternative distance metrics, implement weighted voting, add a faster neighbour index (KD-tree or BallTree), or generate descriptors with correlated features so you can further study the effect of scaling and feature selection.

---

**Full case study, build steps and evaluation checks:** [KNN algorithm implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0007/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
