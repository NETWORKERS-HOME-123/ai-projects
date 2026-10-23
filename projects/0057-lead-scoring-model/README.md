# Lead scoring baseline and recency comparison

## What this project does

This project demonstrates a small, reproducible pipeline for lead scoring using synthetic enquiry histories. It trains a calibrated logistic regression baseline (probabilistic scores) and compares that ranking to a simple recency-based queue. The exercises show how to measure precision at staff capacity, whether groups defined by a sensitive attribute are covered by the top-ranked items, and whether a minimum service standard (recent enquiries) still receives attention even when model scores are low.

## Data

The data are synthetic but designed to mimic enquiry records: numeric features correlated with a binary follow-up outcome, a numeric recency field (days since enquiry) and a protected/sensitive attribute included in the generation so we can demonstrate exclusion from modelling and simple subgroup checks. This is a small stand-in dataset for teaching the pipeline; it is not production data.

## How to build it

1. Prepare the data — generate synthetic enquiry histories with a bounded conversion window and create a protected attribute; exclude the protected attribute from training.
2. Build the experiment — fit a logistic regression baseline and calibrate using CalibratedClassifierCV; also prepare a recency queue that ranks by most recent enquiries.
3. Evaluate the result — compute AUC and Brier score, measure precision at staff capacity (top 10% by default), compare to recency, report subgroup coverage and a simple minimum-service check for very recent enquiries.
4. Define a reviewable output — produce a staff queue that includes model score, recency, the reason flags (recent override), and an opt-out flag simulated for a small fraction.

## How to run

Run the project script. The main entry is run(seed: int = 42) -> dict which returns numeric metrics. To run as a script: python project.py

## How to test

Install requirements and run pytest. The test suite checks that the run() output contains expected keys, numeric ranges, a conservative performance floor on precision at capacity and deterministic results for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| auc | 0.8529 |
| brier | 0.1231 |
| precision_at_capacity | 0.5000 |
| precision_recency | 0.2600 |
| subgroup_coverage_ratio | 1.2500 |
| min_service_rate | 0.0870 |

## Limitations

This is a teaching exercise with synthetic data. The generation is intentionally simple and does not capture real enquiry semantics, temporal drift, selection bias from staff actions, or realistic opt-outs. The protected attribute is synthetic and shown only to illustrate how to exclude sensitive inputs and to compute simple subgroup coverage; no fairness mitigation is implemented. Calibration is a single technique demonstration and would need deeper validation in deployment.

## Next steps

- Replace synthetic data with real, consented enquiry histories and a clearly defined conversion window.  
- Add time-based validation (temporal split) and logging for drift.  
- Audit historical label bias and consider counterfactual labelling or randomized follow-ups to reduce feedback loops.  
- Add constrained optimisation or fairness-aware re-ranking if subgroup gaps are unacceptable.

---

**Full case study, build steps and evaluation checks:** [Lead scoring model on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0057/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
