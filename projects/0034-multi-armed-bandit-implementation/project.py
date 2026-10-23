import json
from typing import List, Dict
import numpy as np


def epsilon_greedy(horizon: int, true_rates: np.ndarray, epsilon: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    k = len(true_rates)
    counts = np.zeros(k, int)
    values = np.zeros(k, float)
    actions = np.empty(horizon, int)
    rewards = np.empty(horizon, int)
    for t in range(horizon):
        if rng.random() < epsilon:
            a = rng.integers(0, k)
        else:
            a = int(np.argmax(values))
        r = int(rng.random() < true_rates[a])
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        actions[t] = a
        rewards[t] = r
    return {"actions": actions, "rewards": rewards, "counts": counts, "values": values}


def ucb1(horizon: int, true_rates: np.ndarray, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    k = len(true_rates)
    counts = np.zeros(k, int)
    values = np.zeros(k, float)
    actions = np.empty(horizon, int)
    rewards = np.empty(horizon, int)
    # initialise by playing each arm once
    for a in range(k):
        if a >= horizon:
            break
        r = int(rng.random() < true_rates[a])
        counts[a] += 1
        values[a] = r / counts[a]
        actions[a] = a
        rewards[a] = r
    for t in range(k, horizon):
        total_counts = np.sum(counts)
        ucb_values = values + np.sqrt(2 * np.log(max(1, total_counts)) / np.maximum(1, counts))
        a = int(np.argmax(ucb_values))
        r = int(rng.random() < true_rates[a])
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        actions[t] = a
        rewards[t] = r
    return {"actions": actions, "rewards": rewards, "counts": counts, "values": values}


def _simulate_policy(policy_fn, name: str, true_rates: np.ndarray, horizon: int, seeds: List[int], **kwargs) -> Dict:
    runs_actions = []
    runs_rewards = []
    runs_counts = []
    for s in seeds:
        out = policy_fn(horizon=horizon, true_rates=true_rates, seed=s, **kwargs)
        runs_actions.append(out["actions"])
        runs_rewards.append(out["rewards"])
        runs_counts.append(out.get("counts", np.zeros(len(true_rates), int)))
    actions = np.vstack(runs_actions)
    rewards = np.vstack(runs_rewards)
    counts = np.vstack(runs_counts)
    return {"name": name, "actions": actions, "rewards": rewards, "counts": counts}


def _evaluate_runs(sim: Dict, true_rates: np.ndarray) -> Dict:
    actions = sim["actions"]
    rewards = sim["rewards"]
    runs, horizon = actions.shape
    optimal_arm = int(np.argmax(true_rates))
    optimal_reward = true_rates[optimal_arm]
    cumulative_regret = np.cumsum(optimal_reward - rewards, axis=1)
    total_regret = cumulative_regret[:, -1]
    fraction_optimal = (actions == optimal_arm).mean(axis=1)
    return {
        "mean_total_regret": float(np.mean(total_regret)),
        "mean_fraction_optimal": float(np.mean(fraction_optimal)),
        "mean_action_counts": list(np.mean(sim["counts"], axis=0).astype(float)),
    }


def run(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    true_rates = np.array([0.20, 0.50, 0.75, 0.60])
    horizon = 2000
    n_runs = 30
    base_seeds = [int(x) for x in rng.integers(0, 10_000, size=n_runs)]
    eg_sim = _simulate_policy(epsilon_greedy, "epsilon_greedy", true_rates, horizon, base_seeds, epsilon=0.1)
    ucb_sim = _simulate_policy(ucb1, "ucb1", true_rates, horizon, base_seeds)
    eg_eval = _evaluate_runs(eg_sim, true_rates)
    ucb_eval = _evaluate_runs(ucb_sim, true_rates)
    # Non-stationary: swap two arms at halfway
    change_point = horizon // 2
    eg_runs = []
    ucb_runs = []
    for s in base_seeds:
        rng_run = np.random.default_rng(s)
        first_half = int(rng_run.integers(0, 1_000_000))
        second_half = first_half + 1
        out1 = epsilon_greedy(horizon=change_point, true_rates=true_rates, epsilon=0.1, seed=first_half)
        changed = true_rates.copy()
        changed[0], changed[2] = changed[2], changed[0]
        out2 = epsilon_greedy(horizon=horizon - change_point, true_rates=changed, epsilon=0.1, seed=second_half)
        actions = np.concatenate([out1["actions"], out2["actions"]])
        rewards = np.concatenate([out1["rewards"], out2["rewards"]])
        eg_runs.append({"actions": actions, "rewards": rewards})
        out1 = ucb1(horizon=change_point, true_rates=true_rates, seed=first_half)
        out2 = ucb1(horizon=horizon - change_point, true_rates=changed, seed=second_half)
        actions = np.concatenate([out1["actions"], out2["actions"]])
        rewards = np.concatenate([out1["rewards"], out2["rewards"]])
        ucb_runs.append({"actions": actions, "rewards": rewards})
    eg_actions = np.vstack([r["actions"] for r in eg_runs])
    eg_rewards = np.vstack([r["rewards"] for r in eg_runs])
    ucb_actions = np.vstack([r["actions"] for r in ucb_runs])
    ucb_rewards = np.vstack([r["rewards"] for r in ucb_runs])
    eg_nonstat = {"name": "epsilon_greedy_nonstationary", "actions": eg_actions, "rewards": eg_rewards, "counts": np.zeros((n_runs, len(true_rates)))}
    ucb_nonstat = {"name": "ucb1_nonstationary", "actions": ucb_actions, "rewards": ucb_rewards, "counts": np.zeros((n_runs, len(true_rates)))}
    eg_nonstat_eval = _evaluate_runs(eg_nonstat, true_rates)
    ucb_nonstat_eval = _evaluate_runs(ucb_nonstat, true_rates)
    result = {
        "policies": ["epsilon_greedy", "ucb1"],
        "stationary": {"epsilon_greedy": eg_eval, "ucb1": ucb_eval},
        "nonstationary": {"epsilon_greedy": eg_nonstat_eval, "ucb1": ucb_nonstat_eval},
        "mean_fraction_optimal_epsilon_greedy": eg_eval["mean_fraction_optimal"],
    }
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
