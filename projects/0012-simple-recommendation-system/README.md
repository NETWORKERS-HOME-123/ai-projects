# Simple user-based recommender

## What this project does

This small project teaches a user-based collaborative filtering pipeline using a compact, permissioned user–item rating table. It shows how to compute user–user cosine similarity, exclude the target user, aggregate neighbour preferences to score unseen items, and fall back to popularity for cold-start users. The implementation reserves a subset of observed ratings as a held-out test set so you can evaluate ranking quality with precision@k and catalogue coverage. There is also a simple popularity baseline to compare against.

## Data

The data are synthetic and built inside the code: a small pandas DataFrame of explicit ratings from a handful of users across several items. The table is permissioned (we only treat entries present in the DataFrame as observed). Missing entries are unknown (not zero). A reproducible held-out split removes a few observed ratings per user for evaluation.

## How to build it

1. Create a virtual environment with Python 3.11+ and install the requirements listed below.
2. Save the project file named project.py and the tests file test_project.py in the same folder.
3. Install dependencies: pip install -r requirements.txt (the requirements file is included).
4. Run the tests (pytest) which also exercise the run(seed) function.

## How to run

Run the script directly: python project.py. It will print a JSON object with the evaluation metrics and a small recommendation list for an example user.

## How to test

Run pytest in the project directory. The tests check that the run function returns the expected keys, that metrics lie in sensible ranges, that a conservative metric floor is met on the held-out precision, and that calling run(42) twice is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| precision_at_3 | 0.2667 |
| baseline_precision_at_3 | 0.1333 |
| coverage | 0.8000 |
| baseline_coverage | 0.6000 |
| example_recommendation_for_u1 | ["i5", "i1"] |
| cold_start_recommendation | ["i2", "i5", "i3"] |

## Limitations

This project demonstrates the mechanics of user-based collaborative filtering but does not deliver a production-quality recommender. The small synthetic dataset means results are noisy. Ratings are sparse by design and we deliberately avoid imputing unknowns with zeros because zero denotes "unknown" here. Simple neighbour aggregation can leave candidate items unscored if no neighbour rated them; the popularity fallback helps but is a blunt instrument. There is no temporal modelling, no hyperparameter search and no fine-grained calibration of rating scales.

## Next steps

Possible extensions include: using weighted neighbour aggregation with multiple neighbours, implementing item-based similarity, adding normalization (mean-centering) of user ratings, learning an explicit latent-factor model, or expanding the dataset so evaluations are more stable and realistic.

---

**Full case study, build steps and evaluation checks:** [Simple recommendation system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0012/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
