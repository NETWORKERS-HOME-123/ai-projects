# Sentiment analysis for social media

## What this project does

This is a small, deterministic teaching project that builds a baseline sentiment classifier for short social-media-style posts. It emphasises an "uncertain" / mixed category to reflect real-world ambiguity in short texts, and demonstrates evaluation by language and by time period. The goal is to show how to prepare minimalised posts safely, build a reproducible pipeline, report aggregate trends and present representative anonymised examples with uncertainty, not to produce a deployed system.

## Data

The dataset is synthetic and deliberately small but designed to mimic challenges found in social media: slang, code-mixing (English + Hindi fragments), abbreviations, and short messages. Each row includes a short post, a language label (short code), a time-period label (e.g. early/late), and a sentiment label in three classes: positive, negative, mixed (uncertain). The README is explicit that the data are synthetic and minimal for teaching; this is not clinical or profiling work and identifiers are not present.

## How to build it

1. Prepare the data — Use permitted, minimised posts constructed inside the code; remove any identifiers and keep examples short and anonymised.
2. Build the experiment — Convert text to TF-IDF (unigrams + bigrams) and train a LogisticRegression baseline with a three-way label space (positive, negative, mixed).
3. Evaluate the result — Use a held-out split and compute macro F1 overall, and broken down by language and by time period. Also compute simple uncertainty indicators from predicted probabilities and present representative anonymised examples.
4. Define a reviewable output — Return a JSON-serialisable dictionary with metrics and example rows showing text, true label, predicted label, and predicted probability; these are aggregated and intended for human review.

## How to run

Run the script as a module or directly. The project exposes a run(seed: int = 42) -> dict function. When executed as a script it prints the returned dictionary as JSON. Everything is deterministic when the same seed is given.

## How to test

This project includes pytest tests. Tests check that run(42) returns the expected keys, that metric values fall in sensible ranges, that the held-out macro F1 meets a conservative floor, and that the main metric is deterministic across repeated calls with the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| overall_macro_f1 | 1 |
| n_train | 31 |
| n_test | 11 |

Also returned (see `python project.py` output): `per_language_macro_f1` (object), `per_period_macro_f1` (object), `examples_uncertain` (5 items).

> **Read the `overall_macro_f1` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This is a teaching baseline using synthetic, compact examples. The dataset is not representative of the full diversity of social-media language, platform-specific conventions, or adversarial/sarcastic uses. Short samples and a small synthetic set limit generalisation: do not use these results to infer private traits or mental state. Manual review and significantly larger, ethically-collected datasets are needed for production work.

## Next steps

Expand with more realistic, consented data; add subword / embedding models; incorporate temporal drift handling; label more nuanced categories and measure inter-annotator agreement; and add bias and fairness checks.

---

**Full case study, build steps and evaluation checks:** [Sentiment analysis for social media on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0064/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
