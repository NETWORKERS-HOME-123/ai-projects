# Word2Vec (small educational SVD approximation)

## What this project does

This small project demonstrates how local word context can produce vector representations (word embeddings). It implements a compact, deterministic pipeline that: tokenises a tiny synthetic corpus, builds a word-context co-occurrence / PMI matrix, applies truncated SVD to obtain low-dimensional embeddings (a well-known approximation related to skip-gram with negative sampling), and queries cosine-nearest neighbours. The README and code emphasise provenance, out-of-vocabulary behaviour and neighbourhood inspection.

## Data

The corpus is a small, synthetic and public-domain-like collection of short sentences created inside the code for teaching purposes. Tokenisation is whitespace+simple punctuation splitting and a minimum frequency threshold filters rare words. The README and code state these choices explicitly; no external data is required or downloaded.

## How to build it

1. Create a Python 3.11+ environment that has numpy, pandas and scikit-learn installed.
2. Save the provided project.py, test_project.py and this README.md in a folder.
3. Install the dependencies: pip install -r requirements.txt (the requirements file is minimal and given with the project).
4. Run the project or the tests as described below.

## How to run

Run the project directly: python project.py

It prints a JSON object with the main metrics and a short explorer output showing nearest neighbours for a few words. The core function is run(seed: int = 42) -> dict which returns the JSON-serialisable results.

## How to test

Run pytest in the project folder: pytest -q

There are deterministic unit tests that check keys, value ranges, a conservative held-out metric floor and determinism for the main metric.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| vocab_size | 14 |
| embedding_dim | 13 |
| main_metric | 0.3333 |
| provenance | synthetic in-code corpus; tokenisation=whitespace+punctuation; min_freq=2; model=PMI+SVD approximation |

Also returned (see `python project.py` output): `neighbour_quality` (object), `explorer` (object).

## Limitations

This project does not implement the original Word2Vec training loop. Instead it uses a small synthetic corpus and a PMI + truncated-SVD pipeline which is a common, pedagogical approximation of skip-gram with negative sampling (Levy & Goldberg). The tiny corpus limits vocabulary coverage and will not capture broad linguistic phenomena. Expect out-of-vocabulary words, limited context diversity and potential corpus-specific biases.

## Next steps

Possible extensions include: replacing the SVD approximation with a lightweight skip-gram negative-sampling optimiser implemented in NumPy, using a larger licensed text corpus, adding subword (character n-gram) handling, and adding evaluation against standard word-similarity benchmarks or downstream tasks.

---

**Full case study, build steps and evaluation checks:** [Word2Vec implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0080/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
