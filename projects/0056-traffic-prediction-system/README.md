# Traffic prediction system

## What this project does

This small teaching project builds a short-horizon traffic-volume baseline using synthetic aggregate road measurements (no personal traces). It demonstrates how to prepare lag features, add simple calendar inputs, compare to a persistence baseline and evaluate performance on held-out future periods and held-out road segments. The model produces planning estimates and flags stale data warnings; it does not perform any automatic traffic-control actions.

## Data

The dataset is synthetic but designed to resemble aggregate segment counts by hour for multiple connected road segments. Each row is an hourly aggregate vehicle count for a segment, with a timestamp, segment id, an incident indicator inserted for some hours, and calendar fields. This is NOT real travel-trace data; it uses only synthetic, aggregate counts generated inside the code.

## How to build it

1. Install the Python dependencies listed in requirements.txt.
2. Open project.py and inspect the run(seed=...) function which builds the synthetic dataset, constructs lag and calendar features, trains a RandomForestRegressor baseline and a persistence comparator, and evaluates held-out times and segments.
3. Run project.py to see a JSON summary of evaluation metrics and warnings about stale data.
4. Use pytest to run the provided tests which check correctness, determinism and a conservative metric floor on the main held-out R2 metric.

## How to run

Run the project directly with Python:

python project.py

It prints a JSON object with evaluation metrics and a small planning table (as JSON-friendly numbers). Change the seed argument in run(seed=...) if you want different random draws.

## How to test

Run the unit tests with pytest in the project directory:

pytest -q

The tests import run from project.py and validate keys, numeric ranges, determinism and a conservative performance floor.

## Results

<!-- RESULTS -->

## Limitations

This project uses synthetic aggregate counts and simple temporal structure. It omits real-world spatial correlations between connected links beyond sharing a base signal generation process, and it does not model routing, lane-level details or complex incidents. The baseline is a short-horizon lag-feature RandomForest and a persistence comparator; more advanced temporal models (e.g. sequence models or graph-based spatiotemporal models) would be needed for production-grade forecasting.

## Next steps

Possible improvements include: adding explicit spatial correlation between neighbouring segments, using probabilistic forecasts, engineering richer incident features (severity, duration), validating on real aggregate traffic counts and extending the horizon and evaluation to multiple days and holidays.

---

**Full case study, build steps and evaluation checks:** [Traffic prediction system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0056/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
