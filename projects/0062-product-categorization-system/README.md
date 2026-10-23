# Product categorisation with TF-IDF

## What this project does

This project demonstrates a compact product categorisation pipeline that maps short catalogue descriptions to a small controlled taxonomy. It trains a TF-IDF text classifier (logistic regression) and adds an explicit fallback path for low-confidence or unseen-category predictions by returning an "UNKNOWN" label and top candidate suggestions suitable for an editor review workflow. The pipeline reports held-out macro F1 and a simple measure of confusion between sibling categories in the taxonomy.

## Data

The code uses a small synthetic dataset of product descriptions generated in-script to stand in for curated catalogue data. Each description is an intentionally varied natural-language phrase (e.g. "lightweight Android smartphone with 64GB") and each row is assigned an editor-style category label from a compact taxonomy with branches such as Electronics, Home and Apparel. This is synthetic and used to illustrate development and evaluation; a real project should replace it with licensed, editor-verified product descriptions and explicit variant grouping prior to splitting.

## How to build it

1. Create the synthetic description corpus and a small hierarchical taxonomy inside the code. Group near-duplicate variants before splitting to avoid leakage.  
2. Split the dataset into training and held-out sets using a fixed random seed and stratification by category.  
3. Train a TF-IDF vectoriser and a LogisticRegression classifier; add a probability threshold to route low-confidence predictions to an UNKNOWN path.  
4. Evaluate on the held-out set: report macro F1 and a sibling-confusion metric; provide top-k candidate categories for editor review.

## How to run

Run the project module as a script (Python 3.11+) or call run(seed) from another program. The script prints a JSON dictionary of numeric metrics and small text values.

## How to test

Use pytest to run the provided tests. They check that run(42) returns expected keys, that metrics are in plausible ranges, that macro F1 meets a conservative floor, and that the run is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| macro_f1 | 0.9845 |
| unknown_rate | 0.0296 |
| sibling_confusion | 0 |

Also returned (see `python project.py` output): `top_k_example` (3 items).

## Limitations

This demonstration uses synthetic text data created inside the code. That means the vocabulary, phrase patterns and category balance are artificial and will not reflect a production catalogue. Also the taxonomy is small and simple; real taxonomies have many nested levels and ambiguous labels. The fallback is a simple low-confidence threshold — in production one would combine this with editorial signals, active learning and candidate-ranking calibrated on real correction logs.

## Next steps

- Replace synthetic descriptions with a licence-approved corpus of editor-verified product descriptions, and group near-duplicate variants before splitting.  
- Expand the taxonomy and evaluate hierarchical losses and per-branch calibration.  
- Introduce an editor-in-the-loop correction UI and record corrections to improve the classifier with incremental training and uncertainty-based sampling.

---

**Full case study, build steps and evaluation checks:** [Product categorization system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0062/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
