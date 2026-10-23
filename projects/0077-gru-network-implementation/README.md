# GRU-like reservoir for simple sequence tasks

## What this project does

This small teaching project implements a tiny sequence experiment that demonstrates gated recurrent behaviour using an explicit GRU forward pass implemented in NumPy. For training I keep the recurrent (GRU) parameters fixed (a simple reservoir approach) and train only a linear readout (logistic regression) on top of the final hidden state. This keeps the code short, deterministic and dependency-light while still showing tensor shapes, hidden-state reset rules and how gating mixes information across time. Two simple baselines are included: (1) a vanilla recurrent unit (no gates) used as a fixed reservoir and (2) a feed‑forward MLP trained on flattened sequences.

The synthetic task is deliberately small and deterministic: binary sequences of variable length where the label depends on the first and last timestep (logical AND). This requires the model to retain information across time steps so you can compare how the gated forward pass and a simple recurrent forward pass handle long-range dependencies.

## Data

All data is synthetic and generated in-code using NumPy. Each sample is a binary sequence (0/1) of fixed length L. The target is the logical AND of the first and the last elements of the sequence. No external datasets are used.

## How to build it

1. Create a fresh Python 3.11+ environment and install requirements (see requirements.txt).
2. Run the project script with Python: python project.py (it prints a JSON dict). The script defines run(seed=42).
3. Inspect results — the script prints held-out accuracies, loss and a short shape/state demo.
4. Run tests with pytest to verify determinism and metric floors.

## How to run

python project.py

The script exposes run(seed: int = 42) -> dict. When run as a script it prints json.dumps(run(), indent=2).

## How to test

Run pytest in the project directory. The provided test file checks (1) expected metric keys in the returned dict, (2) metric ranges (accuracies between 0 and 1), (3) a conservative accuracy floor for the GRU readout and (4) determinism for the main metric.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| gru_test_accuracy | 0.9400 |
| gru_test_loss | 0.2601 |
| rnn_test_accuracy | 0.9960 |
| mlp_test_accuracy | 1 |
| gru_longseq_test_accuracy | 0.7560 |
| runtime_seconds | 0.9625 |
| sequence_length_short | 8 |
| sequence_length_long | 20 |

Also returned (see `python project.py` output): `demo` (object).

> **Read the `rnn_test_accuracy`, `mlp_test_accuracy` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

- This is not training a full GRU with backpropagation-through-time. Instead, the code implements a GRU forward pass (explicit tensor shapes and gating equations) but keeps the recurrent parameters fixed and trains only a linear readout on the final hidden state. This is intentional to keep the project short, deterministic and to focus on tensor shapes, state resets and the effect of gating versus simple recurrence.
- The synthetic task is simple (logical AND of first and last bit) so results do not imply real-world performance on complex sequential tasks.
- No GPU or PyTorch; everything is NumPy + scikit-learn for portability and easy grading.

## Next steps

- Implement full GRU training with gradient-based optimisation (BPTT) using a framework such as PyTorch or JAX.
- Try harder tasks (parity, delayed recall, sequence-to-sequence) and study how training recurrent weights changes dynamics.
- Add visualisations of hidden-state trajectories and gating activations to better explain how information is preserved and gated across time.

---

**Full case study, build steps and evaluation checks:** [GRU network implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0077/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
