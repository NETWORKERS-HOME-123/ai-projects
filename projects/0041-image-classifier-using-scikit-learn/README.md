# Image classifier: scikit-learn digits baseline

## What this project does

This small project demonstrates a baseline image-classification pipeline using scikit-learn's built-in handwritten digits dataset (8×8 grayscale images). It trains a RandomForestClassifier on flattened pixels, evaluates per-digit recall on a stratified held-out set, and constructs a simple digits-only prediction viewer payload that includes the original pixel grid and a visible dataset boundary marker. The README explains limitations: these images are small, built-in digits and do not represent general photographs or external uploads.

## Data

The code uses sklearn.datasets.load_digits (8×8 grayscale images of handwritten digits). This is a small, built-in dataset intended for teaching and experimentation. The project does not accept or test arbitrary uploaded images; it only works with these 8×8 digit images.

## How to build it

1. Create a Python 3.11+ virtual environment.
2. Install the dependencies with: pip install -r requirements.txt
3. Open the project files (project.py and test_project.py) in an editor.
4. Run the main script to print JSON metrics or run the tests with pytest.

## How to run

Run the project script directly: python project.py
It will print a JSON object with accuracy, per-class recall and a small viewer payload.

## How to test

Run the test suite with pytest:

1. pytest -q

The tests assert the presence of expected keys, sensible metric ranges, determinism for a fixed seed, and a conservative accuracy floor.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 0.9694 |
| mean_recall | 0.9691 |
| n_train | 1437 |
| n_test | 360 |

Also returned (see `python project.py` output): `per_class_recall` (object), `top_confused_pairs` (5 items), `example_mistakes` (6 items), `viewer_sample` (object).

## Limitations

These are tiny 8×8 grayscale digit images (scikit-learn's digits). The model and metrics reported here do not validate performance on larger photographs, coloured images, or handwritten images at different resolutions. The pipeline is a baseline using a Random Forest on flattened pixels; it does not perform feature learning or support arbitrary user image uploads.

## Next steps

Possible improvements: add a small image normaliser to handle different input resizes, experiment with simple convolutional feature extractors (outside scikit-learn) or test on larger digit datasets. Also build a lightweight upload-and-resize front-end to convert external grayscale digit images into the 8×8 format the model expects.

---

**Full case study, build steps and evaluation checks:** [Image classifier using scikit-learn on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0041/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
