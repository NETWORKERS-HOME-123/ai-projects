# Multi-armed bandit comparison

## What this project does

This small project implements a simulation framework for multi-armed bandits (MAB), comparing two classic policies: epsilon-greedy and UCB (upper confidence bound). The goal is to demonstrate incremental action-value updates, measure cumulative pseudo-regret, and observe behaviour both in a stationary reward environment and in a changing-reward scenario. The project runs multiple seeded simulations to provide reproducible aggregate metrics and a simple dashboard-style summary of action counts and regret.

## Data

The "data" are simulated Bernoulli reward arms (built-in synthetic data). The code defines several arms with known true success probabilities; those true rates are used only by the simulator to draw rewards and to compute regret and the fraction of optimal actions. No external datasets are required.

## How to build it

1. Install the Python dependencies listed in requirements.txt.  
2. Open the project directory in your Python 3.11+ environment.  
3. Inspect project.py to see the bandit policies and the experiment loop.  
4. Run the project (see "How to run" below) to generate the JSON summary printed to stdout.

## How to run

Run the main module directly with Python 3.11+. The script prints a JSON object returned by run(). Example:

python project.py

The run() function accepts an integer seed argument for reproducibility: run(seed=42).

## How to test

Use pytest. The repository includes test_project.py which runs a few deterministic checks. From the project directory run:

pytest -q

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| mean_fraction_optimal_epsilon_greedy | 0.8468 |

Also returned (see `python project.py` output): `policies` (2 items), `stationary` (object), `nonstationary` (object).

## Limitations

This is a simulated environment with Bernoulli arms only. The policies implemented are basic algorithmic baselines (epsilon-greedy and UCB1) intended for teaching; they are not tuned for every possible bandit problem. The changing-reward scenario in the code is simplistic (one abrupt change mid-run) and therefore does not represent complex non-stationary dynamics.

## Next steps

Possible extensions include: adding Thompson sampling, adding other non-stationary schedules (gradual drift), using contextual bandits, or adding more detailed visualisations (plots) outside the core simulation so results can be explored interactively.

---

**Full case study, build steps and evaluation checks:** [Multi-armed bandit implementation on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0034/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
