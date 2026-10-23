# Correlation analysis tool

## What this project does

This teaching project demonstrates how to compute, inspect and report pairwise linear and rank correlations between numeric variables. It focuses on Pearson (linear) and Spearman (rank) coefficients, and shows simple checks that help avoid misinterpreting coefficients: missing value inspection, outlier sensitivity, bootstrap uncertainty estimates and a quick check for nonlinearity. The project is intentionally small and deterministic so students can run it repeatedly and study each step.

## Data

This project uses the built-in scikit-learn Wine dataset (chemical analysis of wines grown in the same region in Italy). All features are numeric and have documented laboratory units in the original dataset description (for example: alcohol in % by volume, malic_acid in g/L, ash in g/L). The dataset is built-in; no external downloading is required. The code inspects missingness (there are none) and checks for extreme values (outliers) using a z-score threshold.

## How to build it

1. Create a virtual environment with Python 3.11+ and install the requirements file.  
2. Open the project files: project.py contains the analysis pipeline; test_project.py contains pytest checks.  
3. Run the script to produce a JSON-style summary: python project.py.  
4. Run the unit tests with pytest to verify determinism and minimum metric floors.

## How to run

Run the analysis from the repository root: python project.py

The script prints a JSON object with summary metrics (pearson and spearman summaries, bootstrap estimates, sample counts and an outlier-sensitivity check).

## How to test

Install the requirements and run pytest in the repository root. The tests check that the run() output contains the expected keys, that numeric metrics are in valid ranges, that a conservative metric floor is satisfied and that the main metric is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_samples | 178 |
| n_features | 13 |
| missing_cells | 0 |
| n_outlier_cells | 11 |
| rows_with_outlier | 10 |
| pearson_top_abs | 0.8646 |
| pearson_mean_abs | 0.3050 |
| spearman_mean_abs | 0.3092 |
| spearman_top_abs | 0.8794 |
| bootstrap_pearson_mean_abs | 0.3093 |
| bootstrap_pearson_std | 0.0161 |
| pearson_mean_abs_no_outliers | 0.3278 |
| outlier_sensitive_drop | -0.0228 |
| nonlinear_flag | false |
| note | Pearson = linear; Spearman = rank. Correlation != causation. |

## Limitations

This is an educational correlation analysis, not a causal analysis. Correlation coefficients quantify association, not causation — do not infer causal relationships from these numbers alone. The dataset is relatively small and all variables are observational measurements. The pipeline does not produce graphical plots (to avoid extra dependencies); instead it reports numeric summaries that would normally accompany pairwise scatterplots.

## Next steps

Possible extensions for students: add pairwise scatterplots with labelled axes, compute partial correlations controlling for a confounder, use robust correlation measures (e.g. biweight midcorrelation), or expand the bootstrap to produce per-pair confidence intervals and p-values.

---

**Full case study, build steps and evaluation checks:** [Correlation analysis tool on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0018/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
