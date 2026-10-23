# Self-organising map (SOM) demo

## What this project does

This project implements a compact self-organising map (SOM) in NumPy and applies it to a small, low-dimensional synthetic dataset. The SOM learns a 2D grid of prototype vectors from training data using best-matching-unit (BMU) selection and neighbourhood updates with a decaying learning rate and radius. After training, the code evaluates the map using quantisation error and topographic error computed on a held-out test split. The project produces the learned prototype vectors and the neuron assignment for each test sample so you can inspect which neurons represent which region of the input space.

This exercise emphasises algorithmic understanding rather than model complexity: you will implement BMU selection, the neighbourhood function, and the update rule, and examine standard topology-preservation metrics.

## Data

The data are synthetic and generated with scikit-learn's make_blobs function (three numeric features). This is deliberately small and low-dimensional so you can inspect prototypes and assignments easily. The dataset is split into train and test sets; the SOM is trained on the train set and evaluated on the test set.

## How to build it

1. Prepare the data — generate and scale low-dimensional numeric examples and choose a modest grid size (the code uses a 6×6 map by default).
2. Build the experiment — implement BMU selection and update prototypes using a decaying learning rate and neighbourhood radius (Gaussian neighbourhood kernel is used).
3. Evaluate the result — compute quantisation error (average distance to BMU) and topographic error (fraction of samples whose first and second BMUs are not adjacent) on the held-out test split.
4. Define a reviewable output — return prototype vectors and sample assignments (do not interpret cells as fixed natural classes).

## How to run

Install dependencies (see requirements). Then run the project module. The script exposes run(seed: int = 42) and prints a JSON summary when executed as a script.

## How to test

This repository includes pytest tests. Run pytest in the project directory. Tests verify the expected keys are present, metric ranges are sensible, a conservative metric floor is met, and the main metric is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| quantisation_error | 0.1604 |
| topographic_error | 0.1313 |
| seed | 42 |

Also returned (see `python project.py` output): `grid_shape` (2 items), `prototypes` (36 items), `assignments` (99 items).

## Limitations

This implementation is educational and compact. It uses a modest grid and a simple Gaussian neighbourhood; it does not include advanced features like batch training, hexagonal grids, or visualisation. The dataset is synthetic and low-dimensional so results are illustrative rather than representing a large-scale application.

Note: the original starter code sampled Gaussian vectors but did not implement any map-learning steps. This project replaces that with a proper SOM implementation including BMU selection and neighbourhood updates.

## Next steps

Possible extensions include larger or higher-dimensional datasets, additional neighbourhood schedules, hexagonal grid geometry, visualisations of prototypes in 2D, or comparisons with other vector quantisation methods such as k-means.

---

**Full case study, build steps and evaluation checks:** [Self-organizing maps on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0078/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
