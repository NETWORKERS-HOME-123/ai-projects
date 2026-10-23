# Sentiment analysis for movie reviews

## What this project does

This project demonstrates a reproducible baseline for binary sentiment classification of short movie reviews. It trains a TF-IDF text representation and a Logistic Regression classifier using only training text for vocabulary learning, evaluates held-out performance (macro F1 and accuracy), and produces a small review-summary panel that highlights negation, sarcasm and mixed-sentiment examples for inspection.

The dataset used here is synthetic: small, handcrafted movie-style sentences are generated to illustrate typical phenomena (negation, sarcasm, mixed opinions, duplicates) and to give students a safe, deterministic playground for experimentation. The goal is a clean, educational pipeline rather than a production-ready model.

## Data

All examples are created inside the code. The generator produces positive and negative movie-review-like sentences with controlled variations for negation ("not good"), sarcasm ("loved... said no one"), and mixed opinions ("great visuals but weak plot"). Duplicates are removed before splitting, so the held-out set contains distinct sentences.

## How to build it

1. Ensure Python 3.11+ and the packages listed in requirements are installed.
2. Run the project file to train and evaluate the TF-IDF + Logistic Regression pipeline: python project.py
3. The script fits the vectoriser on training text only, evaluates on a held-out test split and prints JSON metrics and a brief review panel.
4. Use pytest to run the unit tests that verify reproducibility, key metrics and a conservative performance floor.

## How to run

Run the project directly: python project.py

The script exposes a run(seed: int = 42) -> dict function that returns a JSON-serialisable dictionary of metrics and example panels. The main block prints this dictionary as formatted JSON.

## How to test

Install pytest and run: pytest -q

The test suite imports run from project.py and checks that the output contains expected keys, that metrics are in sensible ranges, that a conservative macro F1 floor is met, and that the main metric is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| num_examples | 21 |
| num_train | 15 |
| num_test | 6 |
| accuracy | 0.8333 |
| macro_f1 | 0.7778 |

Also returned (see `python project.py` output): `panel_inspection` (3 items), `review_panel` (5 items).

## Limitations

This project uses a small synthetic dataset created for teaching. It is not representative of real movie-review corpora: vocabulary is limited, linguistic diversity is small, and examples are intentionally simplified to illustrate phenomena. Because of the synthetic source, absolute performance numbers are not meaningful for real-world deployment. The model is a simple baseline (TF-IDF + Logistic Regression) and will not handle complex sarcasm or nuanced opinions present in large, real corpora.

## Next steps

1. Replace the synthetic data with a realistic labelled dataset (for example, a properly licensed movie-review corpus) and keep a strict de-duplication step before splitting. 2. Add more sophisticated text features or pretrained embeddings. 3. Experiment with class weights and calibration for probabilistic outputs. 4. Build a small web UI for the review-summary panel to support user corrections and dataset updates.

---

**Full case study, build steps and evaluation checks:** [Sentiment analysis for movie reviews on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0042/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
