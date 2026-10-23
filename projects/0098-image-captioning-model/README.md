# Image captioning model (toy)

## What this project does

This small teaching project demonstrates a toy image-captioning pipeline using only scikit-learn and the built-in digits images (8x8 grayscale). It connects a simple image encoder (a RandomForestClassifier operating on flattened pixels) to a tiny, deterministic caption decoder that produces short template captions from the predicted digit label plus a small, explainable visual attribute. The goal is pedagogical: to show how an image representation can be connected to a language output and how to evaluate and check for hallucinations or missing objects in captions.

## Data

This project uses the sklearn.datasets.load_digits built-in dataset (8x8 grayscale images of handwritten digits 0–9). This is a synthetic/small stand-in and not photographic imagery. Ground-truth captions are synthetic templates generated from the digit label and a simple image-derived attribute ("stroked" vs "filled") so the project demonstrates mechanics rather than real-world captioning.

## How to build it

1. Install the Python requirements listed in requirements.txt.
2. Open project.py and inspect the run(seed: int = 42) function to see data preparation, model training and the captioning decoder.
3. The pipeline splits the built-in digits into train/test with train_test_split(random_state=seed).
4. A RandomForestClassifier (random_state=seed) is trained on flattened pixels and used to produce predicted captions by a deterministic template decoder.

## How to run

Run project.py directly with Python 3.11+; it prints a JSON dictionary of metrics. Example:

python project.py

The main entrypoint is run(seed: int = 42) -> dict.

## How to test

This repository includes pytest tests in test_project.py. Run:

pytest -q

Tests check the returned keys, numeric ranges, determinism and a conservative metric floor on the held-out caption token F1.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| digit_accuracy | 0.9600 |
| caption_token_f1 | 0.9943 |
| num_train | 1347 |
| num_test | 450 |
| model | RandomForestClassifier |
| decoder | deterministic-template |

> **Read the `caption_token_f1` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

- This is a toy demonstration using 8x8 handwritten digit images; it is not a photograph captioning system. The ground-truth captions are synthetic templates derived from the digit label and a trivial image statistic.
- The decoder is a deterministic template generator, not a learned language model; it cannot produce diverse natural language or generalise to real photographic content.
- Human review is still essential for real captioning tasks to detect hallucinations, missing objects, or inappropriate inferences; this project only demonstrates automated checks.

## Next steps

- Replace the template decoder with a small sequence model trained on a real caption dataset (requires appropriate data and licences).
- Swap the encoder for a convolutional neural network (e.g. torchvision models) and fine-tune jointly with a decoder (requires PyTorch and photographic datasets).
- Add human-in-the-loop checks and a stricter suite of automatic metrics (BLEU/ROUGE/SPICE) when moving to real captions.

---

**Full case study, build steps and evaluation checks:** [Image captioning model on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0098/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
