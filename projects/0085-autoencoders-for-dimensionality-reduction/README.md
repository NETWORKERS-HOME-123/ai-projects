# Autoencoders for dimensionality reduction

## What this project does

This beginner project implements a simple bottleneck autoencoder (with a single hidden latent layer) using NumPy and trains it on the sklearn digits dataset. The aim is to learn nonlinear compressed representations and compare them to PCA: we evaluate held-out reconstruction error and a downstream classification task using the compressed features. The project demonstrates how to export encoder features for multiple latent sizes, compares autoencoder and PCA reconstructions, and highlights information lost by compression.

## Data

We use the built-in sklearn digits dataset (8x8 greyscale images, 10 classes). The images are flattened to numeric vectors and normalised to [0, 1]. Data subsets are created with independent train, validation and test splits.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Open project.py and inspect the run(seed=...) function — it contains the whole experiment and is deterministic when seed is fixed.
4. Run python project.py to execute the pipeline and print JSON metrics.

## How to run

Run: python project.py

The script prints a JSON dictionary of metrics and comparisons for each latent dimension tried. The encoder/decoder is implemented with NumPy so no additional deep-learning framework is required.

## How to test

Run pytest in the project directory. The test suite imports run from project.py and verifies the presence and ranges of expected metrics, a conservative metric floor on the held-out classifier accuracy for one latent size, and determinism of the main metric.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| dataset | digits_builtin |
| model | simple_numpy_autoencoder |
| ae_clf_acc_8 | 0.7583 |

Also returned (see `python project.py` output): `latent_dims` (3 items), `results` (object).

## Limitations

This project uses a very small autoencoder built from scratch (single hidden bottleneck) and trains with plain gradient descent; it is not optimised for production performance. The digits dataset is small and constrained; results may differ on larger or more complex images. The autoencoder architecture is intentionally simple for teaching: deeper networks, convolutional layers or more advanced optimisers would usually give better reconstructions. Comparison is limited to PCA and a simple logistic classifier for the downstream task.

## Next steps

- Replace the NumPy autoencoder with a framework implementation (PyTorch/TensorFlow) and add convolutional encoders for images.
- Try more latent sizes and deeper encoder/decoder stacks; add early stopping and better optimisers.
- Evaluate on larger image datasets and expand downstream tasks (e.g. clustering, retrieval).

---

**Full case study, build steps and evaluation checks:** [Autoencoders for dimensionality reduction on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0085/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
