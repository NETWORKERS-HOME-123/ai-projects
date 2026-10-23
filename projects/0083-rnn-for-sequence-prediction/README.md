# RNN next-step prediction (numpy)

## What this project does

This small teaching project implements a simple recurrent neural network (RNN) in pure numpy to perform next-step sequence prediction on synthetic sequences. It demonstrates how hidden state carries information across timesteps, and compares the learned RNN to a simple last-value baseline. A short "sequence replay" is provided so you can inspect where the model succeeds or fails when long dependencies are present.

## Data

The data are synthetic sequences generated inside the code. Each sequence is created by a simple autoregressive rule combining a slow sinusoid and a dependence on the value 3 timesteps earlier, plus Gaussian noise. This creates a task where information from multiple timesteps back can be useful — a small stand-in for long-dependency sequence problems.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Open project.py to inspect the data generation, the small numpy RNN implementation and the evaluation steps.
4. Run the script to train and evaluate the model; it prints a JSON summary.

## How to run

Run the script directly with Python 3.11+. The script exposes a function run(seed: int = 42) that returns a JSON-serialisable dict of numeric metrics and a small text replay. Running the file prints the run output as JSON.

## How to test

Run pytest in the project directory. Tests check that run(42) is deterministic, returns the expected keys, that numeric metrics are sensible and that the main held-out metric meets a conservative floor.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| r2 | 0.9772 |
| mse | 0.0173 |
| baseline_mse | 0.0456 |
| sequence_replay | t	prev	pred	true
0	0.502	0.408	0.573
1	0.573	0.634	0.409
2	0.409	0.594	0.432
3	0.432	0.546	0.167
4	0.167	0.265	0.033
5	0.033	-0.002	-0.068
6	-0.068	-0.214	-0.352
7	-0.352	-0.506	-0.703
8	-0.703	-0.846	-0.777
9	-0.777	-0.984	-0.926
10	-0.926	-1.078	-1.096
11	-1.096	-1.160	-1.364
12	-1.364	-1.305	-1.255
13	-1.255	-1.248	-1.265
14	-1.265	-1.196	-1.263
15	-1.263	-1.142	-1.115
16	-1.115	-1.004	-1.001
17	-1.001	-0.855	-0.966
18	-0.966	-0.762	-0.826
19	-0.826	-0.626	-0.488 |
| model | simple_numpy_rnn |

## Limitations

This is a teaching toy: the RNN is implemented in numpy and trained with a simple SGD loop (no advanced optimisation, no regularisation, no batching beyond full-sequence updates). The synthetic data is small and artificial, so numerical results will not generalise to real-world time series. The model is intentionally simple so students can read and reason about the recurrent updates and backpropagation through time.

## Next steps

Possible next steps include: replacing the numpy RNN with a framework implementation (e.g. PyTorch), adding mini-batching and better optimisers, experimenting with gated recurrent cells (LSTM/GRU) to mitigate vanishing gradients, and evaluating on more complex synthetic tasks or real sequential datasets.

---

**Full case study, build steps and evaluation checks:** [RNN for sequence prediction on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0083/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
