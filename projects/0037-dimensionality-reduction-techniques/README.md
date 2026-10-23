# Dimensionality reduction: PCA vs Isomap

## What this project does
This small project compares a linear projection (Principal Component Analysis, PCA) with a nonlinear embedding (Isomap) on a documented teaching dataset. It evaluates what each method preserves by inspecting (1) reconstruction error where that is defined (PCA), (2) neighbourhood preservation (k-nearest neighbour overlap), and (3) downstream predictive quality when a simple classifier is trained on the reduced 2D representations. The goal is pedagogical: to show differences between linear and nonlinear dimensionality reduction and to measure how those differences affect simple tasks.

## Data
I use the built-in scikit-learn wine dataset (load_wine). It has 13 numeric chemical measurements for three wine classes. The dataset is small and documented; no external downloads are needed. Features are standardised before applying dimensionality reduction.

## How to build it
1. Prepare the data — load sklearn.datasets.load_wine and standardise features with sklearn.preprocessing.StandardScaler. Use a stratified train/test split with a fixed seed.
2. Build the experiment — fit PCA (2 components) and Isomap (2 components, n_neighbors=5) on the training set with fixed parameters and seeds where applicable.
3. Evaluate the result — for PCA compute reconstruction mean squared error on the test set (using inverse_transform); for both methods compute k-nearest-neighbour overlap (neighbourhood preservation) between high-dimensional and low-dimensional spaces; train a simple LogisticRegression classifier on original features and separately on each 2D embedding and evaluate held-out accuracy.
4. Define a reviewable output — return a small JSON-like dictionary with the numeric metrics and short text explaining what each metric means; this is printed when the module is run.

## How to run
Run the module as a script: python project.py. It prints a JSON object with the metrics.

## How to test
This repository includes pytest tests. Run: pytest -q. Tests check keys, sensible metric ranges, a conservative metric floor on PCA 2D accuracy, and determinism of the main metric when run with the same seed.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| pca_recon_mse | 0.4541 |
| knn_overlap_pca | 0.1649 |
| knn_overlap_isomap | 0.2116 |
| acc_original | 0.9815 |
| acc_pca2d | 0.9444 |
| acc_isomap2d | 0.8889 |

Also returned (see `python project.py` output): `explanations` (object).

## Limitations
The wine dataset is small and low-dimensional (13 features); reducing to 2D is aggressive and loses information. Isomap is a manifold method that preserves local geodesic structure; PCA preserves global linear variance. The downstream classifier is a simple LogisticRegression — it is only an illustrative probe of utility, not a comprehensive benchmark. Neighbourhood overlap is a simple proxy for local structure preservation and has its own limitations.

## Next steps
Extend to larger or higher-dimensional datasets (for example digits or custom data), try other nonlinear embeddings (e.g. t-SNE, UMAP), evaluate different classifiers, or measure other preservation criteria (trustworthiness, continuity, mutual information).

---

**Full case study, build steps and evaluation checks:** [Dimensionality reduction techniques on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0037/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
