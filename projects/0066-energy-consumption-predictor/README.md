# Energy consumption predictor

## What this project does

This project trains a simple, reproducible regression baseline to forecast hourly energy consumption for a single aggregate meter. The pipeline demonstrates building lag features, using an as-issued weather input (a forecast column that is not replaced by future true observations), and comparing a trained regression model against a operational baseline: same-hour-last-week demand. The project evaluates chronological mean absolute error (MAE), peak-period errors and holiday-period errors, and emits lightweight warnings for missing-meter readings and stale weather inputs. The project uses a synthetic but realistic hourly dataset created inside the code (see Data section).

## Data

The data are synthetic and generated inside the project using numpy and pandas to mimic permissioned aggregate meter readings with calendar context and an as-issued hourly temperature forecast. The dataset contains hourly timestamps for a year plus two weeks, a true consumption value, a forecast temperature column (the as-issued weather used in production), and holiday markers. A small fraction of meter readings and forecast timestamps are marked to trigger the missing-meter and stale-input warnings.

## How to build it

1. Create a Python 3.11+ virtual environment (recommended). Example: python -m venv venv && source venv/bin/activate
2. Install dependencies: pip install -r requirements.txt
3. Inspect project.py to see the data-generation, lag-feature creation, model training and evaluation logic.
4. Run the pipeline with python project.py to print the JSON results.

## How to run

Run the project script directly. The script defines run(seed: int = 42) and the main block prints the output JSON:

python project.py

## How to test

Tests are provided with pytest. To run the tests:

pytest -q

The tests check presence of keys, reasonable metric ranges, determinism and a conservative main-metric floor.

## Results

<!-- RESULTS -->

## Limitations

- The dataset is synthetic and simplified: it mimics calendar effects, a single as-issued weather forecast column and occasional missing data, but is not a replacement for real operational meter histories or detailed weather models. 
- The regression model used is a linear Ridge baseline with lag features. It is suitable as a starting point but does not replace more advanced methods needed for large, multi-site deployments.
- The forecast temperature column is synthetic. In real deployments you must ensure you pass only the as-issued forecast to the model at inference time; here we simulate that behaviour.
- The read-only chart is a simple textual summary, not a visual plotting output.

## Next steps

- Replace the synthetic data with permissioned historical meter and forecast weather inputs.
- Add probabilistic forecasting or quantile losses to capture uncertainty.
- Extend to multi-site models and incorporate additional exogenous variables (operational schedules, building types).
- Replace the textual chart with an image or interactive dashboard for load planning and operator review.

---

**Full case study, build steps and evaluation checks:** [Energy consumption predictor on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0066/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
