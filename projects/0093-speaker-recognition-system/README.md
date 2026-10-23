# Project 0093: Educational speaker recognition prototype

## What this project does

This small teaching project demonstrates a closed-set speaker-recognition pipeline using synthetic, opt-in audio-like signals. It shows how to: generate a simple multi-session dataset that mimics different speakers and recording sessions, extract lightweight spectral features, train a classifier baseline, and add an explicit unknown-speaker rejection policy based on prediction confidence. The goal is educational: to teach feature extraction, evaluation across sessions, and how to measure false accepts and rejects. This prototype is not suitable for any real biometric use.

## Data

The data are synthetic signals generated inside the code to mimic voice-like differences: each "speaker" is created by summing a few sinusoids with speaker-specific frequency harmonics, plus session-specific noise and small amplitude variation. There are multiple sessions per speaker to enable cross-session evaluation. This is intentionally synthetic and small so no real voice data or private biometric information is used.

## How to build it

1. Create a virtual environment with Python 3.11+ and install dependencies listed in requirements.txt.
2. Place project.py and test_project.py in the project folder (both are provided).
3. Run the example to produce metrics: python project.py
4. Run tests with pytest to verify determinism and metric floors.

## How to run

Run the pipeline locally with: python project.py
The script prints a small JSON object of evaluation metrics computed on a held-out test split.

## How to test

Run the test suite with: pytest -q
The tests check that the run function is deterministic, returns the expected keys, metrics are in valid ranges, and the held-out known-speaker accuracy meets a conservative floor.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| overall_accuracy | 0.1000 |
| known_speaker_accuracy | 0.8000 |
| unknown_reject_rate | 0.8750 |
| false_accept_rate | 0.2500 |
| threshold | 0.7380 |
| holdout_session | 2 |

## Limitations

This project uses synthetic, non-speech audio-like signals and does not contain real voices, transcripts or any biometric data. The recognition task is a didactic closed-set classification with a simple rejection rule; it is not robust across real microphones, languages, or acoustic conditions. Do not use this prototype for access control, identification in the wild, or any deployment involving personal data.

## Next steps

Possible extensions include replacing synthetic signals with a properly consented small speech corpus, extracting richer features (MFCCs or learned embeddings), introducing more realistic noise and reverberation, and testing more principled open-set/thresholding strategies. Always follow privacy and consent best practice when working with real voice data.

---

**Full case study, build steps and evaluation checks:** [Speaker recognition system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0093/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
