# Variational Autoencoder (PCA-based teaching VAE)

## What this project does
This teaching project demonstrates the core ideas of a variational autoencoder (VAE) using only scikit-learn and numpy. It implements a simple encoder that produces a latent mean and log-variance, a reparameterisation-style sampling step, and a decoder. Training here is not a gradient-based optimisation — instead the encoder and decoder are instantiated using PCA on the digits image dataset to keep the example small, deterministic and easy to inspect while still illustrating the reconstruction and KL terms in a VAE objective.

This is intentionally a pedagogical, intermediate-level exercise to teach the mathematical pieces of a VAE (mu, logvar, reparameterisation, reconstruction and KL terms) and to inspect latent usage and held-out reconstructions. It is not a substitute for a full deep-learning implementation with backpropagation.

## Data
This project uses the built-in sklearn digits dataset (8x8 greyscale images, integer values 0–16). The data come from sklearn and are split into train and held-out test subsets inside the code. The dataset is small and licensed for research/education via scikit-learn.

## How to build it
1. Create a virtual environment with Python 3.11+.
2. Install dependencies: pip install -r requirements.txt
3. Open project.py to inspect the pipeline: data split, PCA encoder, sampling, PCA decoder and metric computations.
4. Run the pipeline with python project.py to print JSON metrics to stdout.

## How to run
Run the script directly. It prints a JSON object of metrics for the held-out set:

python project.py

The run function is deterministic when given the same seed.

## How to test
Install pytest (already in requirements) and run:

pytest -q

The tests check output keys, sensible numeric ranges, determinism and a conservative metric floor on the main reconstruction metric.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| explained_variance | 0.2300 |
| reconstruction_mse | 0.0497 |
| kl_mean | 9.4540 |
| elbo | -9.5037 |
| latent_dim | 16 |

## Limitations
- This example is not a learned neural VAE: it uses PCA for encoder/decoder to keep dependencies limited to numpy/scikit-learn. That means gradients, optimisation schedules and deep network representation power are absent.
- The KL term is computed assuming a diagonal Gaussian approximate posterior and a standard normal prior; the encoder's log-variance is estimated from PCA coefficient variances rather than learned per-sample via backprop.
- Reconstructions and latent samples are illustrative rather than high-fidelity image generations.

## Next steps
- Replace PCA encoder/decoder with small neural networks and train with an optimiser (PyTorch or TensorFlow) to implement an end-to-end VAE.
- Implement annealing for KL or beta-VAE variants to study posterior collapse.
- Visualise latent traversals and held-out reconstructions to inspect latent usage and disentanglement.

---

**Full case study, build steps and evaluation checks:** [Variational autoencoder on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0079/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
