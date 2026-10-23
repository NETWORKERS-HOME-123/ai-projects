# Naive Bayes spam classifier

## What this project does

This small project teaches how to build a simple, reproducible Naive Bayes text classifier for spam detection. It demonstrates safe splitting to avoid leakage, fitting a CountVectorizer on training text only, training a MultinomialNB baseline, and computing operational metrics that matter for a review workflow: spam recall (how many spam messages are caught) and legitimate-message false positives (how many ham messages are incorrectly flagged). The data in this project is a small synthetic but realistic stand-in created in code so students can focus on preprocessing and evaluation without external downloads.

## Data

This project uses a small synthetic dataset of short text messages generated in code (so there is no external data download). The generator produces two classes: "spam" messages that contain common advertising/urgent tokens and "ham" (legitimate) messages with ordinary conversational tokens. The dataset is intentionally modest so it runs quickly for students while remaining useful for practising vectorisation and evaluation.

## How to build it

1. Create the synthetic labelled messages in Python (this project does that inside project.py). Do not include any real personal data.  
2. Split into train and test with a fixed random_state so experiments are reproducible.  
3. Fit sklearn.feature_extraction.text.CountVectorizer on the training messages only; inspect the most common tokens.  
4. Fit sklearn.naive_bayes.MultinomialNB on the training counts and evaluate on the held-out test split.

## How to run

Run the main script: python project.py. It will print a small JSON summary of evaluation metrics (accuracy, spam recall, ham false positive rate, a simple majority baseline, and the top tokens learned from training).

## How to test

Run pytest in the project directory: pytest -q. Tests check that the run function returns the expected keys, metrics lie in reasonable ranges, the spam recall exceeds a conservative floor, and that the result is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 1 |
| spam_recall | 1 |
| ham_false_positive_rate | 0 |
| majority_accuracy | 0.5800 |
| n_train | 300 |
| n_test | 100 |

Also returned (see `python project.py` output): `top_tokens` (10 items).

> **Read the `accuracy`, `spam_recall` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses a synthetic dataset generated in code. That makes it suitable for learning and reproducible exercises, but it is not a substitute for evaluation on larger real-world corpora with varied language and privacy requirements. The simple Multinomial Naive Bayes used here is a baseline; real systems often combine models with human review, throttling and additional metadata to reduce legitimate-message loss.

## Next steps

- Replace the synthetic dataset with a responsibly curated real dataset and repeat the experiments.  
- Try TF-IDF and other simple features, compare with MultinomialNB and logistic regression.  
- Build a review queue simulator that surfaces low-confidence predictions for human inspection.

---

**Full case study, build steps and evaluation checks:** [Naive Bayes classifier for spam detection on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0006/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
