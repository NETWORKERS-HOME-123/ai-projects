# Emotion recognition from text

## What this project does

This beginner project demonstrates a small, deterministic pipeline for classifying short text into a set of emotion labels (joy, sadness, anger, fear, neutral, ambiguous). The code trains a text classifier (TF–IDF vectoriser + Logistic Regression) on a synthetic dataset that simulates written expressions and stylistic variation. The pipeline also preserves and returns ambiguous examples for human review rather than producing person-level psychological profiles.

## Data

The data in this project is synthetic and constructed inside the code to stand in for a labelled emotion corpus. It intentionally includes clear emotion-bearing sentences, stylistic variations (slang, punctuation, short forms) and a small set of ambiguous sentences to demonstrate how ambiguous annotations can be preserved for review. The README and code are explicit that this is not a validated, licensed corpus and the dataset is only for teaching and experimentation.

## How to build it

1. Ensure Python 3.11+ and the listed requirements are installed.
2. Put project.py and test_project.py in the same folder.
3. (Optional) Open project.py to inspect the synthetic data, training, evaluation and the ambiguous-example list.
4. Run the project to produce JSON results (see next section).

## How to run

Run the script directly with Python. It will print a JSON dict of metrics and reviewable outputs, for example:

python project.py

The run(seed) function is deterministic when the same seed is supplied.

## How to test

This repository includes pytest tests. Run them with:

pytest -q

Tests check that run(42) returns the expected keys, that numeric metrics are in sensible ranges, that a conservative metric floor is met, and that the main metric is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| macro_f1 | 1 |
| n_train | 135 |
| n_test | 45 |
| classification_report |               precision    recall  f1-score   support

   ambiguous       1.00      1.00      1.00         7
       anger       1.00      1.00      1.00         7
        fear       1.00      1.00      1.00         8
         joy       1.00      1.00      1.00         8
     neutral       1.00      1.00      1.00         8
     sadness       1.00      1.00      1.00         7

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
 |

Also returned (see `python project.py` output): `per_label_f1` (object), `ambiguous_examples_for_review` (7 items).

> **Read the `macro_f1` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses a small, synthetic dataset created in code for teaching. It is not a substitute for a licensed emotion-labelled corpus collected with annotation guidelines and measured inter-annotator agreement. Performance numbers here do not generalise to real-world data. The model predicts sentence-level labels; it does not infer internal mental states or diagnoses.

## Next steps

Possible extensions: replace the synthetic data with a validated corpus (recording annotator disagreement), support multi-label annotations or probabilistic outputs for ambiguous examples, add cross-validation and error analysis across language styles, and create a small review interface for human annotation correction.

---

**Full case study, build steps and evaluation checks:** [Emotion recognition from text on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0088/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
