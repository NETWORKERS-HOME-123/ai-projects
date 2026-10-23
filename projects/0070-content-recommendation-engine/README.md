# Content recommendation engine

## What this project does

This small teaching project builds and evaluates a content recommendation pipeline that starts from collaborative filtering and adds freshness, diversity and a cold-start fallback. The code produces personalised ranked lists for users from a compact synthetic content catalogue. It compares the recommender against a simple editorial baseline using ranking recall, topic coverage and a separate new-user (cold-start) test. The project is deterministic and intended for students to see how small, well-scoped additions to a recommender can change output quality.

## Data

The data in this project is synthetic but realistic in structure: a small catalogue of items with topic labels and publication dates, and a set of voluntary user ratings (1-5). The catalogue and ratings are generated inside the code using numpy and pandas; no external downloads are used. The catalogue provides the metadata required to implement freshness and topic diversity policies.

## How to build it

1. Prepare the data — The code constructs a catalogue of items with topics and publication dates and generates voluntary user ratings. Missing ratings are recorded as NaN (unknown), not as dislike.
2. Build the experiment — Start with user-based collaborative similarity (cosine on mean-imputed ratings) and add a freshness boost for recent items, a simple diversity re-ranking (a greedy MMR-like penalty on repeated topics) and a cold-start fallback that uses popularity combined with recency.
3. Evaluate the result — The pipeline holds out one rating per user to evaluate ranking recall@5 on unseen items, measures topic coverage in the top-5 list, and runs a separate cold-start evaluation by fully hiding one user.
4. Define a reviewable output — The run() function returns ranked item lists (IDs) and numeric metrics. A non-personalised editorial baseline (popularity + recency) is included for comparison.

## How to run

1. Install the requirements (see requirements.txt).
2. Open project.py and read the run(seed=...) function to see the pipeline and parameters.
3. Run python project.py to execute the pipeline; it prints JSON metrics and sample recommendations.
4. Inspect the returned metrics and recommendation snippets.

## How to test

1. Install the requirements: numpy, pandas, scikit-learn, pytest.
2. Run pytest in the project directory. Tests check the presence and ranges of reported metrics, a conservative metric floor for the held-out recall, and determinism of the main metric when run with the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| recall_at_5 | 0.5000 |
| topic_coverage_at_5 | 0.8750 |
| new_user_recall | 0 |
| baseline_recall_at_5 | 0 |
| sample_user | U1 |

Also returned (see `python project.py` output): `sample_recommendations` (5 items).

## Limitations

This is a compact educational pipeline. The synthetic dataset is small and simplified; results do not reflect a production-scale environment. The diversity re-ranking is a basic greedy approach and may not match more advanced constrained optimisation techniques. Freshness is a simple additive score and does not account for long-term user preferences for older content.

## Next steps

Possible extensions include learning-to-rank approaches that combine signals, session-aware freshness, user-level diversity preferences, item-side embeddings (content features), and a better cold-start model that uses side information about users or items.

---

**Full case study, build steps and evaluation checks:** [Content recommendation engine on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0070/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
