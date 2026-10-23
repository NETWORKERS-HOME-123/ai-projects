# Simple educational chatbot (RNN from scratch)

## What this project does

This teaching project implements a tiny sequence-to-sequence recurrent neural network (RNN) trained from scratch with NumPy to demonstrate how encoder–decoder models work for short question–answer pairs. The demo is deliberately small and constrained: it targets a narrow educational topic (basic arithmetic explanations and study tips). The model learns a mapping from short input prompts to short safe replies, and includes an explicit fallback reply when the model is uncertain.

The aim is pedagogical: to show tokenisation, start/end/pad tokens, teacher forcing during training, simple SGD optimisation, and a retrieval baseline for comparison. This is not a production chatbot and the README explains limitations.

## Data

The dataset is synthetic and embedded in the code. It contains a small set of curated question–answer pairs on a harmless educational topic (simple arithmetic facts and study advice). Conversation-level splitting (train / held-out) is used so no held-out pair is seen during training.

Because the dataset is tiny and synthetic, we focus on clarity of implementation and deterministic behaviour rather than high performance.

## How to build it

1. Ensure Python 3.11+ and the required packages are installed (see requirements).  
2. Run the script project.py to train the tiny RNN and evaluate on the held-out set.  
3. The script trains a small encoder–decoder RNN (NumPy) and also evaluates a simple retrieval baseline.  
4. The script prints a JSON report of held-out metrics and sample outputs.

## How to run

Run:

python project.py

This prints a JSON object with metrics and some example held-out inputs, the model replies and the retrieval baseline replies.

## How to test

Install pytest and run:

pytest -q

The tests check the returned JSON keys, that metrics are in sensible ranges, a conservative metric floor on held-out exact-match fraction, and determinism for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| train_size | 7 |
| test_size | 3 |
| exact_match_frac | 1 |
| token_accuracy | 1 |
| retrieval_exact_frac | 1 |

Also returned (see `python project.py` output): `examples` (3 items).

> **Read the `token_accuracy` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is a tiny educational demo. The dataset is synthetic and very small, so the model cannot generalise beyond the narrow prompts it was trained on. The model is trained with a minimal RNN implementation in NumPy (no deep library) and should be used only for learning how encoder–decoder training works. The fallback reply is used when the model's token-level confidence is low. This project does not provide robust conversational capabilities and deliberately avoids giving consequential advice.

## Next steps

Possible extensions for students: increase dataset size, replace the NumPy RNN with a framework-backed implementation (PyTorch / TensorFlow) for efficiency, add beam search decoding, or explore evaluation metrics such as BLEU or human relevance judgements.

---

**Full case study, build steps and evaluation checks:** [Simple chatbot with RNNs on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0094/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
