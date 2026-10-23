# Music genre classifier

## What this project does

This small teaching project demonstrates a simple pipeline for classifying short audio clips into coarse "genre-like" categories using purely synthetic audio. The aim is to teach feature extraction from raw audio (spectral summaries), how to avoid leakages by splitting data by artist/recording source, and how to evaluate a baseline classifier on held-out artists. The code extracts compact spectral features from generated waveforms and fits a RandomForestClassifier as a baseline.

## Data

The dataset is synthetic and constructed in code: short waveform clips are generated from mixtures of sinusoids plus noise. To simulate recording provenance and avoid artist-level leakage, each clip is assigned an "artist" id and each artist is assigned a single genre label. This is intentionally simplified and is not a substitute for real music recordings; it is used for teaching the pipeline and evaluation considerations.

## How to build it

1. Prepare the data — generate synthetic audio clips in Python and assign genre labels and artist ids. Split artists into train and test sets so all clips from a given artist are held out together.  
2. Build the experiment — extract documented spectral features (zero-crossing rate, spectral centroid, bandwidth, RMS energy and a simple spectral flatness proxy) from each clip.  
3. Evaluate the result — fit a RandomForestClassifier on the training artists and report macro F1 and accuracy on held-out artists.  
4. Define a reviewable output — return metrics and counts and include provenance fields (artist id, clip id) in the pipeline logs produced by the code (no redistribution of real recordings since the data is synthetic).

## How to run

Run the module; it exposes a run(seed: int = 42) -> dict function. When executed as a script it prints JSON with the returned metrics. Example: python project.py

## How to test

Tests are provided with pytest. They verify the expected output keys, metric ranges, a conservative performance floor on the held-out macro F1, and determinism for a fixed seed. Run: pytest -q

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| macro_f1 | 0.7688 |
| accuracy | 0.7708 |
| n_train | 144 |
| n_test | 48 |
| model | RandomForestClassifier |

## Limitations

This project uses synthetic audio and simple spectral summaries. Distinguishing these synthetic mixtures is not equivalent to recognising real musical genre, artist identity or production conditions. Real audio contains complex timbral, rhythmic and production variations, and licensed datasets are needed for realistic evaluation. The pipeline nevertheless teaches important steps: feature extraction, artist-level split, baseline modelling and deterministic testing.

## Next steps

Possible next steps include using a small licensed music dataset with artist metadata, adding mel-frequency cepstral features (MFCCs) computed with a dedicated audio library, experimenting with data augmentation and calibration of class imbalance, and inspecting model errors on clips with mixed-genre content or high noise.

---

**Full case study, build steps and evaluation checks:** [Music genre classifier on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0092/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
