# Resume screening evidence extractor

## What this project does

This small, reproducible project demonstrates how to extract explicitly stated skills from short synthetic resumes and build a reviewable, auditable pipeline that does not perform automatic rejection or ranking. The pipeline focuses on extracting cited evidence (skill mentions) and flags missing information; it trains a simple logistic regression model over text features to predict whether a given skill is explicitly stated, and returns evidence spans so a human reviewer can verify or correct outputs.

## Data

The data are synthetic resumes created inside the code from templated sentences and controlled variations (e.g. different formats and innocuous demographic cue phrases). Each synthetic resume is annotated with ground-truth skill mentions for a small, transparent rubric of job-related skills (Python, SQL, communication, testing). This is deliberately synthetic and small so students can inspect every example and see exact annotated evidence.

## How to build it

1. Prepare the data — run the project code which builds a reproducible synthetic dataset of short resume texts and explicit skill annotations.
2. Build the experiment — extract token features from text, train a one-vs-rest logistic regression classifier per skill with a train/test split (random_state fixed) and also compute a simple rule-based evidence extractor.
3. Evaluate the result — compute per-skill precision, recall and F1 on a held-out test split and present evidence spans the rule-based method found so reviewers can verify matches.
4. Define a reviewable output — the run() function returns JSON-serialisable metrics and a small sample of predicted evidence linked to the source text so a reviewer can edit or remove evidence before any decision.

## How to run

Run the project file directly. It is self-contained and deterministic. For Python 3.11+:

python project.py

The script prints a JSON dict with metrics and a short evidence preview.

## How to test

Tests use pytest. Install the requirements and run:

pytest -q

The tests check that the run() function returns the expected keys, that the metrics lie in sensible ranges, that a conservative performance floor is met on the held-out split, and that results are deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|


Also returned (see `python project.py` output): `metrics` (object), `evidence_preview` (8 items).

## Limitations

This project uses synthetic, templated resumes created for teaching and debugging. Real resumes are longer, more diverse and contain formatting (PDFs, images) that this small pipeline does not handle. The logistic regression is a simple baseline; it can miss paraphrases or implicit skill mentions. The pipeline deliberately avoids using demographic cues as features; it shows evidence spans for human review rather than producing automated rejections or rankings.

## Next steps

Possible next steps include: increasing data diversity using consented real examples, using a more powerful NLP featuriser, adding a user interface for human reviewers to correct or delete evidence, measuring robustness to paraphrase and formatting changes, and running fairness audits across controlled demographic variations (without using such cues for prediction).

---

**Full case study, build steps and evaluation checks:** [Resume screening system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0069/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
