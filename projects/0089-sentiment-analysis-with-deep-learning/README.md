# Sentiment comparison: MLP vs TF-IDF baseline

## What this project does

This teaching project demonstrates a small, reproducible comparison between a simple neural classifier (a small multilayer perceptron) and a TF-IDF + logistic regression baseline for binary sentiment classification. It is designed for beginner students to learn about vocabulary construction from the training split, evaluation on held-out data, calibration (Brier score), and simple inference-cost measurement. The code is deterministic given a seed and produces reviewable per-sample predictions to support qualitative inspection.

## Data

The dataset is synthetic but constructed to emulate a sentiment corpus with explicit negation cases and short, review-like sentences. Sentences are generated from templates and word lists inside the code so there is no external data download. This is intentionally small so the project runs quickly on student machines and stays within the allowed Python packages.

## How to build it

1. Create a virtual environment with Python 3.11 or newer.
2. Install the requirements: pip install -r requirements.txt (file contents given in this project).
3. Run the project to train the models and print results: python project.py
4. Run the tests: pytest -q

## How to run

Run python project.py. The script exposes a run(seed: int = 42) -> dict function; running the file prints the JSON result of run(). The run returns metrics for the neural model (MLPClassifier on TF-IDF features) and the TF-IDF + LogisticRegression baseline, along with simple per-sample review suggestions.

## How to test

Run pytest in the project directory. The test suite verifies:
1) The run function returns the expected keys.
2) Metrics are within sensible ranges (0.0–1.0 where appropriate).
3) The main held-out metric (macro_f1_neural) exceeds a conservative floor to ensure the pipeline trains correctly.
4) Determinism: run(seed) is repeatable.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| macro_f1_neural | 0.8730 |
| macro_f1_tfidf | 0.8322 |
| brier_neural | 0.1151 |
| brier_tfidf | 0.1740 |
| infer_time_ratio_mlp_over_baseline | 0.9601 |
| train_time_neural | 0.2513 |
| train_time_tfidf | 0.0096 |
| n_test | 24 |

Also returned (see `python project.py` output): `review_samples` (5 items).

## Limitations

This project uses synthetic text generated from templates, so it is only a small stand-in for real sentiment corpora. The MLPClassifier used here is a simple feed-forward model trained on bag-of-words TF-IDF features rather than modern transformer models; that choice keeps dependencies minimal and the runtime low for educational purposes. Because the data is synthetic and compact, reported scores are indicative of the pipeline behaviour rather than generalisable performance on real-world data.

## Next steps

Possible next steps for students: replace the synthetic dataset with a licensed sentiment dataset, replace the MLP with a deep sequence model (using a framework such as PyTorch) or a pretrained tokenizer + embedding pipeline, evaluate more detailed calibration and cost metrics, and add qualitative error analysis on negation and compositional examples.

---

**Full case study, build steps and evaluation checks:** [Sentiment analysis with deep learning on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0089/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
