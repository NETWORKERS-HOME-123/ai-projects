# Speech-to-text (synthetic classroom demo)

## What this project does

This small teaching project demonstrates a reproducible, deterministic pipeline that emulates a speech-to-text experiment using fully synthetic audio. It is a pedagogical stand-in for an actual licensed ASR model: the code simulates resampling, segmentation and simple acoustic features, trains a small supervised classifier to map short synthetic audio clips to short transcripts, and evaluates word and character error rates (WER and CER) on a held-out split. The goal is to teach the experimental steps (data preparation, documented resampling/segmentation, evaluation by condition, and producing editable transcripts with timestamps) while being explicit about limitations.

## Data

All data are synthetic and generated in code. The project creates short synthetic "utterances" from a small vocabulary of names and numbers (for example "alice", "bob", "twenty three") by synthesising simple tone-based audio patterns and adding optional noise or mild time-stretch to simulate recording conditions (clean / noisy / accented-like). Each synthetic audio clip is paired with a ground-truth transcript. This is intentionally small and artificial so students can inspect every part of the pipeline easily.

## How to build it

1. Create the synthetic dataset in memory: multiple short clips, each with a transcript and a recording condition label (clean, noisy, stretched).
2. Resample and segment each clip in code (documented, deterministic resampling via numpy operations); extract simple spectral-band features from the FFT magnitudes.
3. Train a small supervised classifier (RandomForest) with a held-out test split (random_state controlled) to predict phrase IDs, then map IDs back to transcripts.
4. Compute evaluation metrics (word error rate and character error rate) on the held-out set, and produce per-clip editable transcripts with timestamps and condition metadata.

## How to run

1. Install dependencies: see requirements.txt.
2. Run the project module: python project.py
3. The script prints a JSON report of held-out metrics and a small preview of transcripts.

## How to test

1. Install the test dependency (pytest) in the same environment as the requirements.
2. Run pytest in the project directory. The tests check expected metric keys, metric ranges, a conservative performance floor on the held-out accuracy, and determinism for the main metric.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| n_samples | 168 |
| n_test | 42 |
| n_classes | 7 |
| word_accuracy | 1 |
| wer | 0 |
| cer | 0 |

Also returned (see `python project.py` output): `transcript_preview` (6 items).

> **Read the `word_accuracy` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project uses fully synthetic, tone-based audio and a small classifier to emulate transcription. It does not use a real or licensed ASR model, and therefore cannot represent real speech acoustics, co-articulation, phonetics, or real-world recording variability. The outputs are not real transcripts from human speech; the demo is intentionally simplified so every step is inspectable. Important items such as handling proper names, numbers with context, silence detection or punctuation are simulated and should be reviewed by humans in any real deployment.

## Next steps

1. Replace the toy recogniser with a real licensed ASR model (outside this repo) and compare metrics.
2. Add more realistic synthetic data (concatenated phonemes, room reverberation) or use consented recorded audio with transcripts.
3. Implement per-token confidence scores and an editable transcript UI that preserves retention controls and timestamps.
4. Measure WER/CER per recording condition and add human review workflows for names and numbers.

---

**Full case study, build steps and evaluation checks:** [Speech-to-text converter on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0099/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
