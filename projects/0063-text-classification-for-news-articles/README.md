# Text classification for news articles

## What this project does

This small teaching project demonstrates a multiclass news-topic classifier built with a TF‑IDF representation and a linear classifier. It simulates a simple news corpus with publisher labels and timestamps, deduplicates syndicated copies, performs a temporal train/test split and a held‑out‑publisher evaluation, and reports macro F1 and accuracy. The project also shows how to inspect category‑specific vocabulary and returns editable topic suggestions for sample articles. The data here are a compact synthetic stand‑in created in code to illustrate the pipeline and evaluations — they are not real news articles.

## Data

The data are synthetic, generated in the project code to resemble short news‑style snippets across four topics (politics, sports, technology, business). Multiple small variations and deliberate duplicated (syndicated) copies across publisher IDs are included so the code demonstrates deduplication and a held‑out‑publisher evaluation.

## How to build it

1. Prepare the data — generate short synthetic news snippets with topic labels, publisher IDs and timestamps; deduplicate syndicated copies before splitting.  
2. Build the experiment — convert text to TF‑IDF features (unigrams and bigrams) and train a Logistic Regression multiclass classifier.  
3. Evaluate the result — compute macro F1 and accuracy on a later temporal split and on articles from a held‑out publisher; inspect top tokens per class.  
4. Define a reviewable output — produce editable topic suggestions (probability scores are not credibility scores; suggestions should be reviewed by a human).

## How to run

Run the project module. The main function is run(seed: int = 42) -> dict. Example (from the command line):

python project.py

This prints a JSON dictionary of metrics and a short sample of editable suggestions.

## How to test

Use pytest to run the test suite. From the project directory run:

pytest -q

The tests check keys, metric ranges, a conservative metric floor, and determinism of the main metric.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| macro_f1_time | 1 |
| accuracy_time | 1 |
| macro_f1_publisher | 1 |
| accuracy_publisher | 1 |
| note | Suggestions are editable labels for review; probabilities are not credibility scores. |

Also returned (see `python project.py` output): `top_tokens_per_class` (object), `sample_suggestions` (3 items).

> **Read the `macro_f1_time`, `accuracy_time`, `macro_f1_publisher`, `accuracy_publisher` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses a small synthetic dataset created in code as a compact stand‑in for news text. The snippets are not real journalism and are too small to represent realistic vocabulary, long‑form writing, or the nuance of real editorial labels. Performance numbers shown here are illustrative for teaching the pipeline and evaluation ideas only. Human review is required before using predictions in any downstream process — model outputs are suggestions, not verified facts or credibility scores.

## Next steps

Possible extensions include replacing the synthetic corpus with a licensed real news dataset, adding stronger deduplication (content fingerprinting), experimenting with richer text preprocessing (stemming, named entity features), and evaluating longer temporal ranges and more publishers to better estimate generalisation.

---

**Full case study, build steps and evaluation checks:** [Text classification for news articles on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0063/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
