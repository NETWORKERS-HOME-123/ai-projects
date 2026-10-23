import json
import numpy as np
from sklearn.model_selection import train_test_split

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.n_states = size * size
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.n_actions = len(self.actions)
    def state_index(self, pos):
        return pos[0] * self.size + pos[1]
    def step(self, pos, action):
        move = self.actions[action]
        new_r = max(0, min(self.size - 1, pos[0] + move[0]))
        new_c = max(0, min(self.size - 1, pos[1] + move[1]))
        new_pos = (new_r, new_c)
        if new_pos == self.goal:
            return new_pos, 1.0, True
        return new_pos, -0.02, False

def run(seed: int = 42):
    rng = np.random.default_rng(seed)
    env = GridWorld(size=4)
    seeds = list(range(100, 200))
    train_seeds, test_seeds = train_test_split(seeds, test_size=0.3, random_state=seed)

    def q_learning_for_seed(s):
        rng_local = np.random.default_rng(s)
        Q = np.zeros((env.n_states, env.n_actions))
        alpha, gamma, eps = 0.5, 0.95, 0.1
        episodes, max_steps = 150, 50
        for ep in range(episodes):
            pos = env.start
            for t in range(max_steps):
                s_idx = env.state_index(pos)
                if rng_local.random() < eps:
                    a = rng_local.integers(env.n_actions)
                else:
                    a = int(np.argmax(Q[s_idx]))
                new_pos, r, done = env.step(pos, a)
                ns_idx = env.state_index(new_pos)
                best_next = float(np.max(Q[ns_idx]))
                Q[s_idx, a] += alpha * (r + gamma * best_next - Q[s_idx, a])
                pos = new_pos
                if done:
                    break
        eval_returns = []
        for _ in range(5):
            pos = env.start
            total_r = 0.0
            for t in range(max_steps):
                s_idx = env.state_index(pos)
                a = int(np.argmax(Q[s_idx]))
                pos, r, done = env.step(pos, a)
                total_r += r
                if done:
                    break
            eval_returns.append(total_r)
        return float(np.mean(eval_returns))

    def random_policy_for_seed(s):
        rng_local = np.random.default_rng(s)
        max_steps = 50
        returns = []
        for _ in range(5):
            pos = env.start
            total_r = 0.0
            for t in range(max_steps):
                a = int(rng_local.integers(env.n_actions))
                pos, r, done = env.step(pos, a)
                total_r += r
                if done:
                    break
            returns.append(total_r)
        return float(np.mean(returns))

    train_results = [q_learning_for_seed(s) for s in train_seeds]
    test_results = [q_learning_for_seed(s) for s in test_seeds]
    random_test = [random_policy_for_seed(s) for s in test_seeds]
    test_mean_return = float(np.mean(test_results))
    random_mean_return = float(np.mean(random_test))
    improvement = float(test_mean_return - random_mean_return)

    def build_Q_for_seed(s):
        rng_local = np.random.default_rng(s)
        Q = np.zeros((env.n_states, env.n_actions))
        alpha, gamma, eps = 0.5, 0.95, 0.1
        episodes, max_steps = 150, 50
        for ep in range(episodes):
            pos = env.start
            for t in range(max_steps):
                s_idx = env.state_index(pos)
                if rng_local.random() < eps:
                    a = rng_local.integers(env.n_actions)
                else:
                    a = int(np.argmax(Q[s_idx]))
                new_pos, r, done = env.step(pos, a)
                ns_idx = env.state_index(new_pos)
                best_next = float(np.max(Q[ns_idx]))
                Q[s_idx, a] += alpha * (r + gamma * best_next - Q[s_idx, a])
                pos = new_pos
                if done:
                    break
        return Q

    Q_seed = build_Q_for_seed(seed)
    replay_states, replay_actions, replay_rewards = [], [], []
    action_names = ["left", "right", "up", "down"]
    pos = env.start
    max_steps = 50
    for t in range(max_steps):
        replay_states.append(pos)
        s_idx = env.state_index(pos)
        a = int(np.argmax(Q_seed[s_idx]))
        replay_actions.append(action_names[a])
        pos, r, done = env.step(pos, a)
        replay_rewards.append(float(r))
        if done:
            break

    failure_states = []
    zero_thresh = 1e-4
    for r in range(env.size):
        for c in range(env.size):
            idx = env.state_index((r, c))
            if all(abs(x) < zero_thresh for x in Q_seed[idx]) and (r, c) != env.goal:
                failure_states.append((r, c))

    return {
        "test_mean_return": test_mean_return,
        "random_mean_return": random_mean_return,
        "improvement": improvement,
        "n_test_seeds": int(len(test_seeds)),
        "replay_states": replay_states,
        "replay_actions": replay_actions,
        "replay_rewards": replay_rewards,
        "failure_states": failure_states,
    }

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
