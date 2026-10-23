# Handwriting recognition (digits) classifier

## What this project does
This small teaching project trains and evaluates an image classifier for handwritten characters using the scikit-learn digits dataset (8x8 images of handwritten digits 0–9). It demonstrates a reproducible training pipeline with a documented character vocabulary, a pseudo "cross-writer" split when writer identifiers are not available, and an editable suggestion-style output that provides top candidate characters and an "Unknown" fallback when the model is uncertain.

## Data
The project uses the built-in sklearn.datasets.load_digits dataset (8x8 grayscale images of digits 0–9). The dataset does not include writer identifiers, so the code simulates writer groups deterministically from image statistics to demonstrate a group-based split that approximates cross-writer evaluation. This is a compromise for the classroom exercise; for real cross-writer validation use a dataset that contains explicit writer IDs.

## How to build it
1. Create a Python 3.11+ virtual environment and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Open project.py and study the run(seed: int = 42) function which builds a scaler + logistic-regression pipeline, and evaluates on a held-out group split.
4. Run the project with python project.py to see JSON metrics and example editable suggestions.

## How to run
Run: python project.py
The script prints a JSON object of metrics and a small preview of suggestions for held-out samples.

## How to test
Run pytest in the project directory: pytest -q
Tests assert presence of expected metrics, ranges, determinism, and a conservative accuracy floor.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| dataset | sklearn_digits_8x8 |
| n_train | 1365 |
| n_test | 432 |
| accuracy | 0.9676 |
| top3_accuracy | 1 |
| per_class_accuracy_mean | 0.9693 |
| unknown_fraction | 0.0139 |

Also returned (see `python project.py` output): `vocabulary` (10 items), `suggestions_preview` (10 items).

> **Read the `top3_accuracy` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations
- The dataset used is sklearn's built-in digits collection (8x8 pixels) which is a small, pre-segmented digits dataset. It is not representative of real handwriting lines or multi-character transcription tasks.
- There are no true writer identifiers in the dataset; the project simulates writer groups from image statistics for pedagogical cross-writer splitting. This is explicitly noted and should be replaced with a dataset that contains writer IDs for rigorous cross-writer evaluation.
- The model is a compact classifier (logistic regression) aimed at reproducibility and speed for learning. It is not designed as production OCR and lacks image augmentation or a modern deep learning architecture.

## Next steps
- Move to a dataset with explicit writer identifiers (for example IAM Handwriting Database) and run true cross-writer experiments.
- Replace the classifier with a CNN-based model (e.g. using PyTorch or TensorFlow) and add image augmentation to improve robustness.
- Extend the pipeline to return editable candidate suggestions integrated with a simple UI and incorporate language priors for real transcription tasks.

---

**Full case study, build steps and evaluation checks:** [Handwriting recognition system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0095/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
