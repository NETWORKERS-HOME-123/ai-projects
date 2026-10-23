# Simple grid-world Q-learning agent

## What this project does

This project implements a small, deterministic tabular Q-learning agent that learns to navigate a tiny grid world to reach a goal. It starts from a supplied stateless bandit idea and extends it to a stateful environment with transitions and delayed reward. The code trains the Q-agent across multiple random seeds, compares its held-out performance to a random policy, and produces a short episode replay with per-step reward explanations.

## Data

All data are synthetic: the environment is a tiny 4x4 grid world implemented in NumPy inside the code. There is a single terminal goal state; step rewards are a small negative step cost and a positive reward for reaching the goal. No external datasets are used.

## How to build it

1. Prepare the data — the grid world is created in code, legal actions (up/down/left/right), rewards and episode termination are defined inside project.py.
2. Build the experiment — a tabular Q-learning agent (epsilon-greedy) is implemented and trained from multiple random seeds.
3. Evaluate the result — seeds are split into training and held-out (test) sets; the held-out mean episode return is computed and compared to a random policy baseline.
4. Define a reviewable output — the run() returns metrics and a short deterministic episode replay (states, actions, rewards) showing agent behaviour and explanations.

## How to run

Run the module directly with Python 3.11+. It prints JSON containing metrics and a short episode replay:

python project.py

There is no external I/O and the run() function accepts a seed argument for determinism: run(seed=42).

## How to test

Install requirements and run pytest in the project directory. The test suite checks that the returned metrics contain the expected keys, metrics lie in sensible ranges, the held-out mean return passes a conservative floor, and the run is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| test_mean_return | 0.9000 |
| random_mean_return | -0.1691 |
| improvement | 1.0691 |
| n_test_seeds | 30 |

Also returned (see `python project.py` output): `replay_states` (6 items), `replay_actions` (6 items), `replay_rewards` (6 items), `failure_states` (0 items).

## Limitations

This example is intentionally small and tabular; it uses a toy 4x4 grid with deterministic transitions and fixed rewards. The simplicity makes it useful for teaching but means results do not generalise to large or continuous state spaces. The Q-learning implementation is basic (constant learning rate, epsilon-greedy) and does not use function approximation or advanced exploration.

## Next steps

Possible extensions include: adding stochastic transitions, larger grids, function approximation (neural networks), decaying exploration schedules, and policy evaluation with more seeds and variance reporting.

---

**Full case study, build steps and evaluation checks:** [Simple reinforcement learning agent on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0032/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
