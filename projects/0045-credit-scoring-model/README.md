# Credit scoring baseline and subgroup audit

## What this project does

This teaching project builds a simple, interpretable credit-scoring baseline using a synthetic credit-like dataset. It demonstrates careful handling of missing values, identification of a proxy feature, and a small subgroup fairness audit. The code trains a logistic regression (an interpretable linear model), evaluates calibration with the Brier score, inspects false-positive rates across groups, and emits a short educational model card describing review and appeal steps. The project is explicitly educational and not suitable for real lending decisions.

## Data

The dataset is synthetic and generated inside the code to mimic common credit-record features (age, income, employment length, credit history length, previous defaults) plus a categorical sensitive attribute used for subgroup analysis. The README documents provenance: the data are synthetic for classroom use only, not drawn from real credit records, and thus cannot represent repayment behaviour reliably. Students should note legal restrictions: any real credit scoring must comply with local financial regulation, data protection laws and non-discrimination rules. The synthetic data intentionally inject missing values and a proxy variable correlated with the sensitive attribute to demonstrate how proxy variables can unintentionally encode group information.

## How to build it

1. Create a virtual environment with Python 3.11+.
2. Install the project requirements: numpy, pandas, scikit-learn, pytest.
3. Run the project module; it will generate the dataset, train the model and print results.
4. Run the tests with pytest to verify determinism and minimum performance.

## How to run

Run the project as a script: python project.py
It prints a JSON object with evaluation metrics and a short model card.

## How to test

Run pytest in the project directory: pytest -q
The tests check that the run function returns the expected keys, that metrics are in sensible ranges, that the main held-out metric meets a conservative floor, and that the run is deterministic with the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| auc | 0.8680 |
| brier | 0.0436 |
| threshold | 0.5000 |
| fpr_region_0 | 0 |
| fpr_region_1 | 0 |
| model_card | Model card - educational baseline. Model: LogisticRegression. Data: synthetic classroom data. This is not suitable for operational lending. Appeal: any adverse outcome described here would require human review, documented reason codes and an appeal process. Reviewers should check feature provenance, missing data handling and proxy correlations (for example the 'proxy' column). |

## Limitations

This project uses synthetic data only and is intended for classroom demonstration. Synthetic records do not capture real repayment dynamics or the institutional, behavioural and collection biases found in operational credit systems. The logistic regression baseline is interpretable but simplistic; production systems require rigorous validation, human oversight, anti-discrimination review, and legal compliance. Do not use the outputs for any actual credit decision-making.

## Next steps

Possible next steps include: replace synthetic data with a properly licensed real dataset; extend to causal and counterfactual analyses; perform larger fairness audits and calibration over score buckets; add thresholds and appeal workflows that involve human reviewers and logging for auditability.

---

**Full case study, build steps and evaluation checks:** [Credit scoring model on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0045/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
