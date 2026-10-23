# GAN-like digits generator

## What this project does

This small educational project implements a toy Generative Adversarial Network (GAN)-style training loop in pure NumPy that learns to generate 8×8 grayscale digit-like images from scikit-learn's built-in digits dataset. The code trains a small generator and discriminator (both simple multilayer perceptrons implemented with NumPy), alternately updating them with SGD. The project emphasises reproducibility, simple diagnostics (fixed-seed sample grids), and checks for diversity and memorisation by computing nearest neighbours to the training set.

## Data

This project uses the scikit-learn built-in load_digits dataset (8×8 grayscale digit images). These images are small, low-resolution digit shapes intended for educational use. Generated images are synthetic and derived from this public dataset; you may use them for learning, demonstration and non-commercial coursework purposes consistent with scikit-learn's data redistribution and citation guidance. This project does not generate or attempt to recreate photographs or identifiable people.

## How to build it

1. Create a Python 3.11+ virtual environment.
2. Install dependencies: pip install -r requirements.txt.
3. Open project.py and read the run(seed=...) function to understand the pipeline.
4. Run the script to train and inspect outputs (see below).

## How to run

Run the script directly: python project.py. The script prints a JSON object with numeric metrics when finished.

## How to test

Run pytest in the project directory: pytest -q. Tests check structure, ranges, determinism and a conservative metric floor on the held-out discriminator accuracy.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| discriminator_accuracy | 1 |
| generator_diversity | 0.4598 |
| mean_nearest_train_dist | 2.0647 |

> **Read the `discriminator_accuracy` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is a toy, educational implementation. The images are 8×8 digits — far smaller and simpler than modern photographic image tasks. The neural nets are small and trained with plain SGD on this tiny dataset: they are intended to teach how adversarial alternating updates can be implemented and inspected, not to produce high-fidelity or general-purpose image generators. No identity or private data is involved.

## Next steps

Possible improvements include using larger convolutional networks, experimenting with different losses (e.g. Wasserstein loss), adding regularisation to reduce memorisation, and visualising training dynamics over checkpoints. Also consider switching to a deep autograd framework (PyTorch/TF) for larger-scale experiments.

---

**Full case study, build steps and evaluation checks:** [GAN for generating simple images on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0086/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
