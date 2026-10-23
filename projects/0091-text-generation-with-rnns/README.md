# Text generation with a tiny RNN (character-level)

## What this project does

This teaching project shows how a very small, character-level recurrent neural network (an Elman RNN implemented in pure numpy) can be trained to predict the next character in short documents and used to generate text by seeded sampling with temperature control. The implementation is intentionally tiny and deterministic so students can read and run it quickly and inspect both held-out loss and generated samples.

## Data

This project uses a deliberately small synthetic corpus defined inside the code (a handful of short public-domain-like paragraphs constructed for the exercise). The corpus is not intended as a realistic language dataset: it exists so students can run and inspect a full next-character training + sampling pipeline without downloads.

## How to build it

1. Create a virtual environment with Python 3.11+ and install requirements.
2. Put the three files from this project (project.py, test_project.py, README.md) in a directory.
3. Activate the environment.
4. Install the Python packages from requirements and run the code.

## How to run

Run the project directly:

python project.py

This prints a JSON object with held-out metrics and three generated samples at different temperatures.

## How to test

Run pytest in the project directory:

pytest -q

This runs deterministic unit tests that check return keys, metric ranges, a conservative floor for the held-out accuracy and determinism of the main metric.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| val_loss | 3.6078 |
| val_accuracy | 0.1574 |
| sample_temperature_0.5 | the w oooooooowile ooooooooooooooooowsin ooooow ooooow woow ooo |
| sample_temperature_1.0 | the om moooadot by ooft w wa by ha oowssig w t oow owkt orirng  |
| sample_temperature_2.0 | thes silominiwhe dyssldocmsapt ofd cmowrin r wothttpafr by t bn |
| vocab_size | 22 |
| epochs | 40 |

## Limitations

This is a tiny educational model implemented in numpy (not using frameworks such as PyTorch). The synthetic small corpus and the small RNN are only suitable for demonstration and learning; performance and linguistic quality are very limited. Generated text may show repetition and memorised short fragments from the tiny corpus. The project also omits practical concerns such as batching efficiency, GPU acceleration, and long-context handling.

## Next steps

Possible next steps for students: replace the synthetic corpus with a larger public-domain text (take care to split documents to reduce leakage), swap the numpy RNN for a proper framework implementation (e.g. PyTorch), add better optimisers, and implement evaluation metrics to detect memorisation and repetition.

---

**Full case study, build steps and evaluation checks:** [Text generation with RNNs on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0091/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
