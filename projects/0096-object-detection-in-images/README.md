# Object detection (synthetic stand-in)

## What this project does

This teaching project demonstrates object-detection concepts using a small, deterministic, synthetic stand-in dataset derived from scikit-learn's digits images. Real object-detection pipelines operate on photos with annotated bounding boxes; here we extract simple foreground bounding boxes from each 8×8 digit image and train lightweight scikit-learn models to predict object class and the single bounding box per image. The aim is to illustrate training, held-out evaluation, IoU-based matching and a basic mean average precision (mAP) calculation without heavy deep‑learning frameworks.

## Data

The data are synthetic: scikit-learn's load_digits (8×8 grayscale images). For each image we derive one bounding box by finding the minimal rectangle around bright pixels. These boxes are a small, approximate stand-in for annotated boxes in real images. Because the original dataset provides only class labels (digit), this project explains that limitation and uses the derived boxes solely for pedagogic detection metrics.

## How to build it

1. Create a Python 3.11+ environment with the requirements file in this project. Install packages using pip install -r requirements.txt.
2. Open project.py and inspect the run(seed: int = 42) function which performs data preparation, training and evaluation.
3. Run the script via python project.py to see JSON metrics printed to stdout.
4. Run tests with pytest to verify determinism and metric floors.

## How to run

Run the pipeline directly: python project.py

It prints a JSON object with evaluation metrics (mAP@0.50, mean IoU, classification accuracy). The pipeline is deterministic when you pass the same seed.

## How to test

Run pytest in the project directory. Tests check that run(42) returns the expected metric keys, that numeric metrics are in sensible ranges and that the main held-out metric meets a conservative floor. They also check run(42) is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| map50 | 0.9091 |
| map75 | 0.9091 |
| mean_iou | 0.9996 |
| class_accuracy | 0.9750 |
| viewer_sample_count | 10 |

Also returned (see `python project.py` output): `viewer_sample` (10 items).

## Limitations

This is a synthetic educational exercise. The dataset is not photographs and the bounding boxes are derived heuristically from tiny 8×8 images — they are not human-drawn annotations. The models used are lightweight scikit-learn regressors/classifiers, not modern convolutional detectors. Consequently metrics here do not validate real-world object-detection performance. The project intentionally avoids any automated safety‑critical actions and outputs only numerical metrics and a simple editable bounding-box viewer (textual coordinates) for review.

## Next steps

- Replace the synthetic boxes with a small licensed dataset that contains real images and human-annotated boxes.
- Swap the scikit-learn models for a deep-learning detector (e.g. a PyTorch Faster R-CNN) and re-evaluate mAP at multiple IoU thresholds.
- Add visualization (image overlays) and per-size analysis to inspect missed small objects more deeply.

---

**Full case study, build steps and evaluation checks:** [Object detection in images on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0096/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
