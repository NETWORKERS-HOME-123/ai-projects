# Image similarity search engine

## What this project does

This project demonstrates a simple image similarity search pipeline using scikit-learn and the built-in digits image collection (8×8 grayscale images). It converts images into compact embeddings (via PCA), builds a nearest-neighbour index, and performs retrieval for a small labelled query set. The project measures retrieval quality with recall@k and mean reciprocal rank and returns ranked provenance information for inspection.

## Data

This project uses the sklearn.datasets.load_digits collection (8×8 greyscale images). To create a realistic retrieval catalogue with duplicates and small appearance variations the code programmatically synthesises duplicate groups by adding tiny intensity noise and scaling to copies of selected images. The query set is a small human-labelled selection drawn from the original images and slightly perturbed to simulate photographed queries.

## How to build it

1. Prepare the data — load the sklearn digits dataset and create a catalogue with duplicated entries and provenance (original index, duplicate id).
2. Build the experiment — compute PCA embeddings for all catalogue images and index them with sklearn.neighbors.NearestNeighbors (euclidean distance).
3. Evaluate the result — run retrieval for a small labelled query set, compute recall@1, recall@5 and mean reciprocal rank, and capture ranked provenance for manual inspection.
4. Define a reviewable output — the pipeline returns ranked neighbour IDs with provenance so a reviewer can view thumbnails and mark irrelevant results.

## How to run

Run the project module. It prints a small JSON object with metrics and sizes, for example:

python project.py

The function run(seed: int = 42) is the programmatic entry point and returns a JSON-serialisable dict.

## How to test

Run pytest in the project directory. Tests verify the returned keys, value ranges, determinism for a fixed seed and a conservative metric floor on recall@5.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| recall_at_1 | 1 |
| recall_at_5 | 1 |
| mrr | 1 |
| catalogue_size | 5391 |
| query_size | 20 |
| method | PCA(32)+NearestNeighbours(brute,euclidean) |
| seed | 42 |

Also returned (see `python project.py` output): `ranked_provenance_sample` (3 items).

> **Read the `recall_at_1`, `recall_at_5` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This demonstration uses very small 8×8 greyscale digit images; they are not representative of natural photos. The embeddings are linear (PCA) and the nearest-neighbour index is an in-memory brute-force index; behaviour on larger or colour image collections would require different descriptors and a scalable ANN index. The synthetic duplicates are simple perturbations, so retrieval metrics here do not ensure similar behaviour on real photographed product or person images.

## Next steps

1. Replace PCA with a pretrained convolutional embedding (outside the restricted library set) for richer semantic similarity.
2. Use data augmentation that mimics real capture conditions (cropping, blurring, background changes).
3. Swap brute-force NN for an approximate nearest neighbour index (FAISS, Annoy) when scaling to larger catalogues.

---

**Full case study, build steps and evaluation checks:** [Image similarity search engine on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0067/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
