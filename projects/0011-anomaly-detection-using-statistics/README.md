# Anomaly detection with rolling baseline and robust thresholds

## What this project does

This project teaches a statistics-first approach to flagging anomalous sensor readings. Using a synthetic time series with a seasonal baseline and injected incidents, the code computes two kinds of thresholds from an initial baseline window: a classic z-score threshold (mean ± k·std) and a robust median/MAD rule. The held-out portion of the series is evaluated using these statistical cutoffs, and simple alert validation is performed against the known incidents. The focus is on counting true/false alerts and measuring precision and recall so students can appreciate the burden of false alarms and the need for human review.

## Data

The data are synthetic and produced inside the code. A sinusoidal seasonal signal plus Gaussian noise models a permitted sensor series. Discrete anomaly spikes and sustained deviations are injected at known timestamps to simulate incidents; these provide ground truth for validation.

## How to build it

1. Ensure Python 3.11+ and the required packages are installed (see requirements). Create a virtual environment if you prefer.  
2. Install dependencies: pip install -r requirements.txt (the file lists numpy, pandas, scikit-learn, pytest).  
3. Open project.py to inspect the synthetic data generation, the baseline selection, and the two thresholding rules (z-score and median/MAD).  
4. Run the pipeline with the provided run(seed) entry point to compute held-out metrics and see flagged periods.

## How to run

Run the module directly: python project.py. That prints a JSON dictionary of metrics for the held-out evaluation. You can also import run from project and call run(seed) from other code.

## How to test

Run pytest in the project directory: pytest -q

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| precision | 1 |
| recall | 0.0370 |
| tp | 1 |
| fp | 0 |
| fn | 26 |
| baseline_mean | -0.0326 |
| baseline_std | 3.5269 |
| baseline_median | -0.0781 |
| baseline_mad_scaled | 4.8984 |
| upper_z | 8.7845 |
| lower_z | -8.8498 |
| upper_mad | 14.6170 |
| lower_mad | -14.7732 |
| held_out_size | 600 |

> **Read the `precision` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is a compact educational example. The synthetic series approximates a seasonal sensor but is much simpler than real-world multi-sensor telemetry. The rules here are univariate and do not model correlated signals. Thresholds and window choices are fixed for clarity; a production deployment would need continuous recalibration, drift handling, and a logged human-review workflow.

## Next steps

Suggested extensions include: (1) compare different baseline window sizes and quantify how quickly the detector adapts, (2) incorporate rolling MAD rather than a single baseline MAD, (3) add a simple human-review log (a labelled buffer) to simulate escalation decisions, and (4) explore small supervised classifiers or time-series models only after careful labelling and review.

---

**Full case study, build steps and evaluation checks:** [Anomaly detection using statistics on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0011/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
