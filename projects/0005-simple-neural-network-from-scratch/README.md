# Simple neural network from scratch

## What this project does

This teaching project implements a tiny feed-forward neural network from first principles using NumPy. The aim is to show how forward propagation, a loss function and backpropagation work in practice for a small classification task. The code trains a one-hidden-layer network on a built-in synthetic binary dataset and compares the network to a simple logistic regression baseline from scikit-learn. A small finite-difference gradient check is included to confirm the hand-derived gradients match numerical approximations.

## Data

The dataset is synthetic and generated with scikit-learn's make_moons function (a classic two-dimensional binary classification toy). The data is small and deliberately simple so students can inspect weights, activations and losses. The dataset is split into a training set and a held-out test set using a fixed random seed.

## How to build it

1. Create a virtual environment with Python 3.11+ and install the requirements listed in requirements.txt.  
2. Open the project file (project.py) to review the NumPy implementation of a single hidden layer neural network, loss and backpropagation.  
3. Run the training pipeline (project.py) which trains the network and computes held-out metrics and a logistic-regression baseline.  
4. Inspect the returned dictionary for training loss, test loss, test accuracy, baseline accuracy and a small gradient-check difference.

## How to run

Run the module directly: python project.py. The module prints a JSON object produced by run(seed=42).

## How to test

The project includes pytest tests. Run pytest in the project directory. Tests check that the run() output has the expected keys, that metric ranges are sensible, that a conservative accuracy floor on the held-out set is met, and that repeated calls with the same seed are deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| train_loss | 0.3185 |
| test_loss | 0.3034 |
| test_accuracy | 0.8333 |
| baseline_accuracy | 0.8333 |
| grad_check_diff | 0.0000 |

## Limitations

This is an educational implementation for understanding fundamentals. It is intentionally minimal: there is no batching beyond a simple SGD loop, no advanced optimisers, no regularisation tuning, and no GPU acceleration. The toy dataset is synthetic and small so the demonstration focuses on learning dynamics rather than production performance.

## Next steps

Possible extensions for students: add additional hidden layers, implement other activation functions, add L2 regularisation, implement mini-batch training and momentum, or visualise decision boundaries and loss curves (not included here to keep dependencies minimal).

---

**Full case study, build steps and evaluation checks:** [Simple neural network from scratch on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0005/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
