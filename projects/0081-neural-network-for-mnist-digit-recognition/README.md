# Neural network for handwritten digit classification (MNIST-scoped)

## What this project does

This small teaching project demonstrates supervised neural classification on handwritten digit images using a compact dense (fully connected) neural network. It trains a multilayer perceptron on a small, built-in digit dataset, holds out validation and test splits, reports per-digit accuracy on the held-out test set, and provides a simple textual inspection of misclassified examples with confidence scores. The goal is to practice tensor/array shapes, loss-driven training, held-out evaluation and cautious interpretation of results.

## Data

This project uses the scikit-learn built-in digits dataset (load_digits). That dataset contains 8×8 grayscale digit images and is used here as a small stand-in for MNIST (28×28) so students can run experiments quickly without downloading external data. The README is honest about this: the data is built-in and smaller than canonical MNIST.

## How to build it

1. Prepare the data — Load the digits dataset and split into train/validation/test sets using fixed random_state so experiments are repeatable.
2. Build the experiment — Standardise pixel values and train a small dense network (scikit-learn's MLPClassifier) with cross-entropy (log-loss) on the training split.
3. Evaluate the result — Use the validation split for lightweight monitoring and report final per-class and overall accuracy on the held-out test split. Inspect a few misclassified examples together with predicted probabilities.
4. Define a reviewable output — Return a JSON-serialisable dictionary of metrics (overall accuracy, per-class accuracy, split sizes and model configuration). The README and code stress that results are MNIST-scoped and not general OCR claims.

## How to run

Run the project file with Python 3.11+. For example:

python project.py

This prints a JSON object with metrics and will run deterministically when you pass the same seed.

## How to test

Use pytest to run the bundled tests:

pytest -q

The tests check that the run function returns the expected keys, that metrics are in valid ranges, that a conservative accuracy floor is met on the held-out test set, and that runs with the same seed are deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| test_accuracy | 0.9750 |
| n_train | 1257 |
| n_val | 180 |
| n_test | 360 |

Also returned (see `python project.py` output): `per_class_accuracy` (object), `model_hidden_layers` (1 items), `misclassified_examples` (6 items).

## Limitations

This project uses the smaller scikit-learn digits dataset as a faithful but downscaled stand-in for MNIST. The model is a densely connected classifier (MLPClassifier) rather than a convolutional neural network, so it does not capture spatial inductive biases that are useful for full-sized MNIST or real OCR tasks. No external data is downloaded, and no claims are made about real-world OCR performance.

## Next steps

Suggested next steps for students:
- Replace the dataset with full MNIST (download and adapt shapes).
- Move from a dense MLP to a small convolutional network (requires a deep learning library).
- Add learning curves, early stopping and simple hyperparameter sweeps on validation split.
- Implement a small visualiser (graphical) for misclassified images in a notebook environment.

---

**Full case study, build steps and evaluation checks:** [Neural network for MNIST digit recognition on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0081/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
