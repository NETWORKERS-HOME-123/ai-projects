import json
import random
from typing import List, Tuple, Dict

import numpy as np

# Simple GA for small 0/1 knapsack

def exhaustive_optimum(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int]]:
    n = len(values)
    best_val = 0
    best_sol = [0] * n
    for mask in range(1 << n):
        sol = [(mask >> i) & 1 for i in range(n)]
        w = sum(wi * si for wi, si in zip(weights, sol))
        if w <= capacity:
            v = sum(vi * si for vi, si in zip(values, sol))
            if v > best_val:
                best_val = int(v)
                best_sol = sol
    return best_val, best_sol


def initialise_population(pop_size: int, n_bits: int, rng: random.Random) -> List[List[int]]:
    return [[rng.choice([0, 1]) for _ in range(n_bits)] for _ in range(pop_size)]


def fitness(ind: List[int], values: List[int], weights: List[int], capacity: int) -> int:
    total_w = sum(w * s for w, s in zip(weights, ind))
    total_v = sum(v * s for v, s in zip(values, ind))
    if total_w > capacity:
        return 0  # infeasible solutions get zero fitness
    return int(total_v)


def tournament_select(pop: List[List[int]], fits: List[int], k: int, rng: random.Random) -> List[int]:
    idxs = rng.choices(range(len(pop)), k=k)
    best = max(idxs, key=lambda i: fits[i])
    return pop[best][:]


def single_point_crossover(a: List[int], b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    if len(a) <= 1:
        return a[:], b[:]
    pt = rng.randrange(1, len(a))
    ca = a[:pt] + b[pt:]
    cb = b[:pt] + a[pt:]
    return ca, cb


def mutate(ind: List[int], rate: float, rng: random.Random) -> List[int]:
    return [bit ^ (1 if rng.random() < rate else 0) for bit in ind]


def run(seed: int = 42) -> Dict:
    rng = random.Random(seed)
    np.random.seed(seed)

    # Problem definition (small, 10 items so exhaustive check is cheap)
    values = [6, 3, 5, 4, 6, 2, 7, 4, 3, 5]
    weights = [2, 1, 3, 2, 4, 1, 5, 3, 1, 2]
    capacity = 15

    n_bits = len(values)
    pop_size = 100
    generations = 200
    tournament_k = 3
    mutation_rate = 0.02
    elitism = 2

    # Precompute exact optimum
    optimal_value, optimal_sol = exhaustive_optimum(values, weights, capacity)

    population = initialise_population(pop_size, n_bits, rng)

    fitness_history: List[float] = []
    best_sol = None
    best_fit = 0

    for gen in range(generations):
        fits = [fitness(ind, values, weights, capacity) for ind in population]
        # record best of generation
        gen_best = max(fits)
        fitness_history.append(float(gen_best))
        if gen_best > best_fit:
            best_fit = gen_best
            best_sol = population[fits.index(gen_best)][:]

        # Stopping rule: reach optimal
        if best_fit >= optimal_value:
            break

        # Elitism: keep top elites
        elites_idx = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)[:elitism]
        new_pop = [population[i][:] for i in elites_idx]

        while len(new_pop) < pop_size:
            parent_a = tournament_select(population, fits, tournament_k, rng)
            parent_b = tournament_select(population, fits, tournament_k, rng)
            child_a, child_b = single_point_crossover(parent_a, parent_b, rng)
            child_a = mutate(child_a, mutation_rate, rng)
            child_b = mutate(child_b, mutation_rate, rng)
            new_pop.append(child_a)
            if len(new_pop) < pop_size:
                new_pop.append(child_b)

        population = new_pop

    # Ensure best_sol is feasible
    if best_sol is None:
        best_sol = [0] * n_bits
        best_fit = 0

    fraction = float(best_fit) / float(optimal_value) if optimal_value > 0 else 0.0

    result = {
        "seed": int(seed),
        "best_solution": list(map(int, best_sol)),
        "best_feasible_fitness": int(best_fit),
        "optimal_fitness": int(optimal_value),
        "fraction_of_optimal": float(fraction),
        "fitness_history": list(map(float, fitness_history)),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
