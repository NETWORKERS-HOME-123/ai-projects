# Image style transfer (toy demonstration)

## What this project does

This project demonstrates a small, deterministic, pedagogical approximation of image "style transfer" using only numpy and scikit-learn built-in data. It is NOT a neural style transfer implementation (no deep network or optimisation loop). Instead it shows how simple statistics of image features (means and variances of filtered responses) can be matched to transfer a coarse "style" while preserving content structure. The code is intentionally small and deterministic so students can read and extend it.

## Data

This project uses the built-in sklearn digits dataset (8×8 greyscale images) as both content and style images. This is a tiny, synthetic stand-in for larger photographs. The README is explicit about this limitation: the digits dataset is not representative of general photographic content and the method here is a simplified statistical transfer (affine match of simple filter responses), not neural feature matching.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. Install the required packages: pip install -r requirements.txt
3. Inspect project.py to see the deterministic pipeline: data selection, feature extraction (simple filters), compute affine transform to match statistics, produce transferred image and metrics.
4. Run the project with python project.py to see the JSON summary printed to stdout.

## How to run

Run:

python project.py

It prints a JSON object with a few numerical metrics and provenance values (means of content/style images). The generated image is returned as a numpy array inside the process (not written to disk) and a small textual summary is printed via the JSON output.

## How to test

Run pytest in the project directory (pytest will run the included test_project.py which checks API shape, metric ranges, determinism and a conservative metric floor):

pytest -q

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| content_index | 1245 |
| style_index | 220 |
| content_image_mean | 0.2773 |
| style_image_mean | 0.3369 |
| content_mse | 0.0001 |
| content_corr | 0.9999 |
| relative_change | 0.0212 |
| alpha | 0.8500 |

## Limitations

This demonstration uses 8×8 greyscale digit images from sklearn as a stand-in for photographs and does not use a convolutional neural network. The style method is a simple affine match of low-level filter statistics; it cannot reproduce complex texture or semantic style transfer. The dataset and algorithm are chosen for pedagogy and determinism rather than visual fidelity.

## Next steps

Potential extensions for students: replace the feature extractor with deep features from a permitted pretrained network (if allowed in another environment), run optimisation in pixel space, support RGB images and larger resolutions, or implement iterative neural-style optimisation using appropriate deep-learning frameworks and licensed pretrained weights.

---

**Full case study, build steps and evaluation checks:** [Image style transfer on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0090/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
