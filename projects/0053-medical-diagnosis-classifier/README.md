# Medical diagnosis classifier

## What this project does

This small teaching project trains an interpretable baseline classifier for a binary medical diagnosis task using a licensed, de-identified teaching dataset included with scikit-learn. The goal is educational: demonstrate careful data handling (explicit missing-data treatment and patient-level separation), train a simple logistic regression baseline, and report clinically-relevant metrics (sensitivity, specificity) together with calibration and uncertainty estimates. The code is intentionally sandboxed and does not provide patient-specific diagnosis or treatment advice.

## Data

This project uses the scikit-learn breast cancer dataset (diagnostic Wisconsin). That dataset provides de-identified, per-patient numeric features and a binary target (malignant/benign). No external data are used and no patient identifiers are exposed.

## How to build it

1. Create a reproducible Python environment with the required packages (see requirements).
2. Open the project file project.py and review the preprocessing: missing-value imputation, scaling, and the logistic regression baseline.
3. Run the pipeline locally to produce held-out metrics and subgroup checks.
4. Inspect results and the README notes about limitations before drawing any clinical conclusions.

## How to run

Run the project module directly. It prints a JSON summary of metrics to stdout.

## How to test

Run pytest in the project directory. Tests check that the run() function returns the expected keys, that numeric metrics are within valid ranges, that a conservative performance floor is met on the held-out split, and that results are deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 0.9650 |
| sensitivity | 0.9245 |
| specificity | 0.9889 |
| brier | 0.0215 |
| ece | 0.0323 |
| subgroup_sensitivity_diff | -0.0816 |
| model | LogisticRegression |

Also returned (see `python project.py` output): `sensitivity_ci` (2 items), `specificity_ci` (2 items).

## Limitations

This exercise is educational. Even though the dataset is a real, de-identified teaching dataset, retrospective performance on this dataset does not establish diagnostic safety or clinical benefit. The model is a simple logistic regression baseline and has had no clinical validation, prospective testing, nor assessment of harms. The pipeline also uses a single held-out test split; a full evaluation would include external validation, robust calibration assessment in new groups, and clinician review.

## Next steps

Suggested next steps for learning: (1) add nested cross-validation and repeated runs to better estimate uncertainty; (2) evaluate external datasets or temporal splits; (3) include clinician adjudication for any ambiguous labels; (4) explore simple explainability methods (feature coefficients, SHAP-lite analyses) and assess fairness across subgroups before considering any clinical translation.

---

**Full case study, build steps and evaluation checks:** [Medical diagnosis classifier on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0053/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
