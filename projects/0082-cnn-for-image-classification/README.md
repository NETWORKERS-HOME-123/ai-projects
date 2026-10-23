# CNN-like feature extractor on sklearn digits

## What this project does

This small teaching project implements a compact convolutional-like feature extractor using pure NumPy on the sklearn built-in 8×8 handwritten digits dataset, then trains a simple classifier (LogisticRegression) on the extracted features. The goal is educational: to show convolution, activation and pooling as explicit array operations, compare to a simple baseline, and report per-class errors and a held-out accuracy.

## Data

The dataset is sklearn.datasets.load_digits (8×8 grayscale images, 10 classes). This is a small built-in dataset appropriate for demonstration only; it is not representative of larger photographic image tasks. No external downloads are used.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Inspect project.py to see the convolution, ReLU and pooling implementations (all NumPy).
4. Run the pipeline with python project.py to produce JSON metrics.

## How to run

Run: python project.py

This prints JSON metrics from run(seed=42) to stdout.

## How to test

Run the pytest test suite: pytest -q

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 0.9289 |
| baseline_accuracy | 0.1022 |
| n_classes | 10 |
| per_class_error_mean | 0.0717 |

## Limitations

This project does not use a deep-learning framework (PyTorch/TensorFlow). Instead it implements convolution, activation and pooling as explicit NumPy array operations for teaching. The dataset is the small 8×8 sklearn digits set; conclusions do not generalise to larger photographic datasets. The classifier used is LogisticRegression on the extracted features, not a trained CNN with learned convolution kernels via backprop.

## Next steps

Possible next steps: implement kernel learning with gradient descent (a tiny framework or use PyTorch), add data augmentation, or visualise learned filters after training in a framework that supports end-to-end learning.

---

**Full case study, build steps and evaluation checks:** [CNN for image classification on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0082/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
