# Document clustering with TF-IDF + K-means

## What this project does

This small educational project demonstrates unsupervised document clustering. It builds a compact synthetic document collection (since we do not use external corpora here), removes duplicate texts, represents documents with TF–IDF (unigrams and bigrams), and compares two grouping methods: K-means on TF–IDF vectors and a simple similarity-based baseline that groups by nearest-neighbour cosine similarity above a threshold. The project computes cluster summaries (representative terms and sizes), a silhouette score for the K-means clusters on a held-out split, and a simple stability measure comparing two different random seeds.

The example also includes simple programmatic operations you would expose in a small UI: rename a cluster label and move a document from one cluster to another (both simulated and returned in the output so reviewers can see editable actions).

This emphasises that the original starter used supervised logistic regression; here we carefully treat the task as unsupervised clustering and report appropriate cluster-quality metrics plus simple stability checks for review.

## Data

The dataset is synthetic and built inside the code from short sentences that mimic four topical groups (technology, health, finance, and travel). A few near-duplicate sentences are intentionally included and removed in preprocessing. The README is explicit that the data is synthetic and small so students can run and inspect everything locally.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. pip install -r requirements.txt (the file lists numpy, pandas, scikit-learn, pytest).
3. Inspect project.py to see the data construction, preprocessing, vectorisation and clustering pipeline.
4. Run project.py to execute the small experiment and view the JSON results.

## How to run

Run the module directly with Python. The script has an if __name__ == "__main__" block which prints a JSON object of results. Example:

python project.py

You can pass a different seed by calling run(seed=123) from other code.

## How to test

Run pytest in the project directory. The test file checks that run(42) returns the expected keys, that numeric metrics lie in sensible ranges, that the main held-out metric (silhouette) exceeds a conservative floor, and that run(42) is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_documents | 16 |
| n_train | 11 |
| n_hold | 5 |
| k | 4 |
| silhouette | 0.3934 |
| stability_agreement | 0.4545 |
| seed | 42 |

Also returned (see `python project.py` output): `cluster_sizes` (object), `top_terms` (object), `baseline_group_sizes` (object), `editable_labels` (object), `example_move_action` (object).

## Limitations

This project uses a small synthetic dataset for reproducibility and classroom speed. Real document collections are larger, noisier and often require more advanced preprocessing (stopword tuning, named-entity handling, subword tokenisation for some languages) and different vector representations (word embeddings, contextual models). Cluster quality here is indicative of the pipeline rather than of real-world deployment.

K-means assumes spherical clusters and requires choosing k. The silhouette score is useful but imperfect for high-dimensional sparse text. Stability is measured here by label agreement across seeds; human review of representative documents is essential in practice.

## Next steps

Possible extensions: larger licensed corpora, experiment with different vectorisers (word2vec, transformer embeddings), automatic selection of k (gap statistic, silhouette grid), interactive labelling/merging UI, and manual curation tools for moving documents between clusters.

---

**Full case study, build steps and evaluation checks:** [Document clustering system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0050/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
