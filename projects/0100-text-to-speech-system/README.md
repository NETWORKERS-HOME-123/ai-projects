# Text-to-speech (Toy synthesis + ML quality predictor)

## What this project does

This teaching project demonstrates principles of text-to-speech (TTS) pipelines while remaining fully offline and deterministic. It contains three parts: (1) a small, explicit text normalisation routine that prepares text for synthesis; (2) a tiny deterministic toy synthesiser that converts normalised text into a short synthetic waveform (sine-based, clearly labelled as synthetic and not a human voice); and (3) a simple machine-learning model that predicts a synthetic "intelligibility" / quality score from numeric features derived from many synthetic examples. The ML model shows how you can evaluate and predict audio-quality-like targets using a held-out split.

This is intentionally a small, educational project — it does not produce real speech or use licensed neural TTS models. It focuses on the engineering steps around TTS: normalisation, controlled synthesis settings, evaluation metrics and determinism.

## Data

All data is synthetic and created inside the code using sklearn's make_regression generator combined with deterministic text-derived features. A handful of example phrases are used to demonstrate the text normalisation and toy synthesis; the ML training data is generated programmatically so no external files or audio recordings are required.

## How to build it

1. Create the synthetic dataset using sklearn.datasets.make_regression with a fixed random seed so results are reproducible. Derive additional text-like numeric features from small example phrases to demonstrate normalisation.  
2. Implement a deterministic toy synthesiser: text -> normalised tokens -> parameters -> short numpy waveform (sine waves). Keep sampling rate and duration controlled by parameters.  
3. Fit a regression model (LinearRegression) to predict a continuous "quality" target on the generated numeric features, using train_test_split with a held-out test set.  
4. Evaluate held-out performance with R2 and mean absolute error, and return those metrics in a JSON-serialisable dictionary.

## How to run

Run the project module directly. It prints a JSON object with the reported metrics and a short example of normalisation + synthesis statistics.

Example (from command line):

python project.py

(No network, no files required.)

## How to test

Use pytest to run the unit tests included in the repository. Tests check that the run(seed) function returns expected keys, that metrics fall into sensible ranges, that the main held-out metric exceeds a conservative floor, and that repeated calls with the same seed are deterministic.

To run tests:

pytest -q

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| r2_train | 0.9927 |
| r2_test | 0.9921 |
| mae_test | 13.8180 |
| example_wave_mean | 0.1812 |

> **Read the `r2_train`, `r2_test` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses a synthetic toy synthesiser (sine waves) and synthetic numeric targets. It does not synthesise real human speech and does not model phonetics, prosody or neural vocoders. The toy audio is intentionally simple so students focus on pipeline steps: text normalisation, controlled synthesis parameters, held-out evaluation and deterministic code. It does not perform voice cloning or use any licensed TTS model; it only demonstrates the surrounding engineering patterns.

## Next steps

- Replace the toy synthesiser with an appropriate open, licensed TTS model and small permitted text prompts.  
- Add a perceptual listening-test pipeline (human annotations) and use those labels as training targets.  
- Extend the normaliser to a full rule-based or learned frontend handling more abbreviations, numbers and named entities.  
- Export labelled synthetic audio with clear metadata and playback controls in a web UI (still avoiding voice cloning).

---

**Full case study, build steps and evaluation checks:** [Text-to-speech system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0100/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
