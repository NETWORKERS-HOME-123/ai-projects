# Image segmentation with colour clustering

## What this project does

This teaching project demonstrates a simple, unsupervised image segmentation pipeline that clusters pixel colour features using K-means. It builds small synthetic colour images (so no external image files are required), fits K-means on a held-out subset of pixels, and predicts cluster assignments for the remainder. The project reports a held-out silhouette score (a clustering quality metric) and K-means inertia. It also returns the chosen cluster count so students can experiment with how the number of clusters affects the segmentation.

## Data

The data are synthetic, small RGB images generated in code using NumPy (no external image files or downloads). Each image contains simple coloured regions (rectangles and circles) arranged to mimic natural regions with clear colour differences. Images are resized by construction to 64×64 pixels and represented in RGB colour space (three channels). This keeps the task lightweight while giving realistic pixel-colour clustering behaviour.

## How to build it

1. Create the synthetic colour images in code (64×64 RGB arrays) with simple shapes and solid colours. No external files are used.  
2. Flatten pixels (rows × cols × channels → N × 3) and normalise colour values to [0, 1].  
3. Split pixels into a train and held-out test set using train_test_split (fixed random_state). Fit K-means on the train pixels and predict labels for the test pixels.  
4. Compute evaluation metrics on the held-out pixels (silhouette score using RGB features and predicted labels, and K-means inertia) and return them in a JSON-serialisable dict.

## How to run

Run the project module as a script. It prints a JSON object of results. Example (from shell):

python project.py

Or import and call the run(seed) function from Python.

## How to test

Use pytest. The provided tests import run() and verify the output structure, value ranges, determinism, and a conservative metric floor on the held-out silhouette score.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| silhouette | 0.7358 |
| inertia | 214.4296 |
| n_clusters | 4 |

## Limitations

- This is an unsupervised pixel-colour clustering demonstration. The clusters are colour/appearance-based regions, not semantic object masks.  
- The images are synthetic and small (64×64) so results do not transfer directly to real photographs or large images without further work.  
- Silhouette score measures compactness and separation in colour space; it does not directly measure boundary accuracy or semantic correctness. IoU is not used because no ground-truth semantic masks are available.

## Next steps

- Replace synthetic data with licensed photographic images and, if available, ground-truth masks to evaluate segmentation with IoU.  
- Add spatial features (pixel coordinates) to help K-means prefer contiguous regions.  
- Explore other clustering algorithms (MeanShift, Agglomerative) and colour spaces (Lab) to compare segmentation quality.

---

**Full case study, build steps and evaluation checks:** [Image segmentation using clustering on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0051/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
