# Simple genetic algorithm for a tiny knapsack

## What this project does

This small project implements a simple genetic algorithm (GA) to solve a toy 0/1 knapsack problem. The GA demonstrates core evolutionary components: population initialisation, fitness evaluation, selection, crossover, mutation and an explicit stopping rule. The code compares the best feasible solution found by the GA with the exact optimal solution found by exhaustive search for a compact item set, and repeats the experiment deterministically across random seeds.

The primary educational goal is to make the evolutionary loop explicit and reviewable: the project returns feasible candidate solutions and a fitness history rather than claiming an exact optimum.

## Data

The problem data are synthetic and defined in the code: a small list of item values and weights with a fixed capacity. Because the set is small (10 items), it is feasible to compute the exact optimum by brute force and use that as a reference.

## How to build it

1. Create a new Python 3.11+ environment and install the requirements listed in requirements.txt.  
2. Save project.py and test_project.py from this project into a directory.  
3. Run the project as a script (python project.py) to see JSON output of the GA run.  
4. Run pytest to execute the automated tests (pytest -q).

## How to run

Execute the script directly: python project.py. The script exposes a run(seed: int = 42) -> dict function and prints JSON summarising the GA run when invoked from the command line.

## How to test

Run pytest in the project directory: pytest -q. Tests check that the run function returns the expected keys, that metrics are in valid ranges, that the GA finds a solution reasonably close to the known optimum (a conservative floor) and that the run is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| seed | 42 |
| best_feasible_fitness | 32 |
| optimal_fitness | 32 |
| fraction_of_optimal | 1 |

Also returned (see `python project.py` output): `best_solution` (10 items), `fitness_history` (3 items).

> **Read the `fraction_of_optimal` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

This project purposefully targets a tiny, synthetic knapsack so the exact optimum can be computed for comparison. The GA implementation is intentionally simple (single-point crossover, basic mutation, tournament selection) and not tuned for large or real-world instances. For larger problems, performance and runtime behaviour will require additional design (advanced operators, constraints handling, parallelism).

Note: the data are synthetic and small; this project teaches algorithm structure and diagnostics rather than offering production-ready solvers.

## Next steps

Possible improvements include: testing other selection/crossover operators, implementing constraint-preserving repair for infeasible offspring, multi-objective extensions, adaptive mutation rates, and scaling up item counts with GPU or parallel evaluation for fitness.

---

**Full case study, build steps and evaluation checks:** [Simple genetic algorithm on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0026/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
