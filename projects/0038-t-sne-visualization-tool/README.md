# t-SNE visualisation toolkit

## What this project does

This small teaching project shows how to generate and compare multiple t-SNE low-dimensional views of a modest dataset. It produces 2D embeddings for several perplexities and random seeds, computes neighbourhood trustworthiness for each view, and evaluates layout stability by comparing pairwise-distance correlations across seeds. The code also records library versions and emits short textual warnings about interpreting global distances in t-SNE layouts.

## Data

This project uses the built-in Iris dataset from scikit-learn (150 rows, 4 numeric features). Features are standard-scaled before computing distances and embeddings. Labels from the dataset are kept only to report how many classes are present and for potential colouring in a visual tool (no plotting in this teaching code).

## How to build it

1. Create a Python 3.11+ virtual environment.
2. Install dependencies: pip install -r requirements.txt
3. Run the project module: python project.py
4. Run tests: pytest -q

## How to run

Execute python project.py. The module prints a JSON object with metadata, per-run metrics (trustworthiness and distance-correlation) and summary statistics including the average trustworthiness.

## How to test

Run pytest in the project directory. The test suite checks that the run function returns the expected keys, that numerical metrics are in valid ranges, that the average trustworthiness exceeds a conservative floor, and that the main metric is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| dataset | iris |
| n_samples | 150 |
| n_features | 4 |
| n_classes | 3 |
| numpy_version | 2.5.4 |
| pandas_version | 3.1.0 |
| sklearn_version | 1.9.1 |
| avg_trustworthiness | 0.9841 |
| warning | t-SNE emphasises local neighbourhoods; global distances and areas are not reliably preserved. Use trustworthiness and stability metrics when interpreting layouts. |

Also returned (see `python project.py` output): `perplexities` (3 items), `seeds` (3 items), `runs` (9 items), `stability` (object).

## Limitations

This project does not produce graphical output. A full interactive visual tool would include linked scatterplots and brushing and require a plotting library or a browser-based UI. t-SNE emphasises local structure; global distances and areas are not reliably preserved and can change with perplexity and random seed. The code therefore reports trustworthiness (local neighbourhood preservation) and distance-correlation (how well pairwise distances are linearly preserved) rather than relying on appearance alone.

## Next steps

Possible next steps include adding interactive linked plots (e.g. with plotly or Bokeh), supporting other datasets, adding UMAP for comparison, and implementing Procrustes alignment for more refined stability comparisons between embeddings.

---

**Full case study, build steps and evaluation checks:** [T-SNE visualization tool on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0038/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
