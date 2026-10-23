# Simple autoencoder with MLPRegressor

## What this project does
This project builds a small autoencoder using scikit-learn's MLPRegressor to learn an encoder–decoder by reconstructing its inputs. The pipeline trains on a held-out subset, compares reconstruction quality to PCA (same latent dimension), and returns quantitative metrics plus a few held-out samples with the largest reconstruction errors. The implementation is intentionally simple and deterministic to make it suitable for teaching and review.

## Data
We use the built-in sklearn.datasets.load_digits dataset (8×8 greyscale images). This is a small image dataset included with scikit-learn and is sufficient for demonstrating an autoencoder workflow without external downloads.

## How to build it
1. Create a Python 3.11+ virtual environment and activate it.
2. Install the required packages: pip install -r requirements.txt
3. Run the project script: python project.py
4. Run the tests: pytest -q

## How to run
Run python project.py. The script prints a JSON dictionary with held-out reconstruction metrics and a few example reconstructions (numerical arrays) for inspection.

## How to test
Run pytest in the project directory. Tests check that the run(seed) function returns the expected keys, that metrics lie in sensible ranges, that the main held-out metric meets a conservative floor, and that run(42) is deterministic.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| dataset | digits |
| n_train | 1437 |
| n_test | 360 |
| latent_dim | 8 |
| test_mse_scaled | 0.2494 |
| pca_mse_scaled | 0.3909 |
| reconstruction_r2 | 0.7084 |

Also returned (see `python project.py` output): `top_errors` (5 items).

## Limitations
This is a minimal educational autoencoder implemented with scikit-learn's MLPRegressor rather than a deep-learning framework. The encoder and decoder are represented implicitly by the MLP's hidden layers and are not exported as separate objects. Using MLPRegressor means we do not get easy layer-by-layer control or GPU acceleration. The dataset is small and built-in; results do not generalise to large image datasets.

## Next steps
Possible extensions include: implementing the autoencoder in PyTorch for explicit encoder/decoder modules; experimenting with convolutional architectures for images; visualising reconstructions as images; and trying different bottleneck sizes and regularisation strategies to explore overfitting versus useful compression.

---

**Full case study, build steps and evaluation checks:** [Simple autoencoder on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0039/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
