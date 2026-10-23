# Transfer learning with a simulated pre-trained backbone

## What this project does

This project demonstrates the idea of transfer learning without using deep libraries. We simulate a pretrained visual backbone by fitting a PCA (principal component analysis) on a source image dataset, then adapt that representation to a new target task (handwritten digits classification). The pipeline trains three heads and compares held-out accuracy:

- A frozen-backbone head: PCA is fitted on a source dataset and kept fixed; a logistic regression classifier is trained on top of the fixed features.
- A limited fine-tune head: we simulate a low-rate fine-tuning by blending the pretrained PCA components with components fitted on the target training data (small blending coefficient), then training a classifier on those features.
- A from-scratch baseline: PCA is fitted only on the target training data and a classifier is trained on those features.

The project also computes a tiny proxy for model size/cost (number of floating parameters used by the linear head) and documents the provenance of the simulated checkpoint, preprocessing and permitted uses in the model card.

## Data

All data are built-in to scikit-learn. We use sklearn.datasets.load_digits (8x8 greyscale images of handwritten digits 0–9). To simulate a realistic pretraining scenario, a separate "source" set is created by adding light Gaussian noise and a small rotation-like transform to a copy of the digits data; this is only a pedagogic stand-in for a licensed pretrained image checkpoint.

## How to build it

1. Install the Python dependencies listed in requirements.txt.
2. Run the project module (python project.py) to produce held-out metrics printed as JSON.
3. Read the model card text printed by the run() function to inspect checkpoint provenance and permitted uses.
4. Run pytest to execute the test suite that verifies determinism and a conservative performance floor.

## How to run

Run the file directly with Python 3.11+: python project.py

The script prints a JSON object with held-out accuracies for the frozen, fine-tuned and scratch baselines and a tiny model-card string describing provenance and preprocessing.

## How to test

Use pytest: pytest -q

Tests check that the returned dictionary has the expected keys, that accuracies are in sensible ranges, that the fine-tuned accuracy meets a conservative floor, and that the run function is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy_frozen | 0.9333 |
| accuracy_finetune | 0.9311 |
| accuracy_scratch | 0.9578 |
| n_components | 30 |
| params_head | 300 |
| model_card | Model card:
Simulated pretrained PCA fitted on a transformed copy of sklearn.datasets.load_digits.
Preprocessing: input pixels are 0-16 greyscale; we centre and project with PCA.
Checkpoint provenance: simulated (no real external checkpoint).
Permitted uses: research and teaching on small-image transfer simulations.
Not for deployment on real image tasks without replacing PCA with a real pretrained CNN and checking licence and data overlap. |

## Limitations

This is a simulation of transfer learning using linear PCA as the "backbone". It is not a substitute for experiments with real pretrained convolutional networks (for example, torchvision models). The simulated checkpoint has different properties to a modern CNN: PCA is linear, uses small 8x8 inputs and does not capture the kinds of features learned by deep visual backbones. The synthetic source-pretraining process (noise and small transform) is a pedagogic stand-in; it does not replicate issues like licence compatibility, dataset overlap or original preprocessing pipelines that matter when using real checkpoints.

## Next steps

- Replace the PCA backbone with a true pretrained CNN when PyTorch/torchvision are available and follow the checkpoint licence and provenance steps.
- Expand the domain-shift study: vary noise/transform severity and inspect per-class confusion to locate systematic domain-shift errors.
- Add a small calibration for cost: FLOPs or inference timing on representative hardware.

---

**Full case study, build steps and evaluation checks:** [Transfer learning with pre-trained models on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0087/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
