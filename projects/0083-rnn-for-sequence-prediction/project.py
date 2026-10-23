import json
from typing import Dict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def generate_sequences(n_seqs=200, seq_len=40, seed=0):
    rng = np.random.RandomState(seed)
    seqs = []
    for i in range(n_seqs):
        t = np.arange(seq_len)
        slow = 0.8 * np.sin(0.2 * t + rng.uniform(0, 2 * np.pi))
        x = np.zeros(seq_len)
        # initialize first three values
        x[:3] = rng.normal(scale=0.1, size=3) + slow[:3]
        for k in range(3, seq_len):
            # dependence on value 3 steps earlier + slow sine + noise
            x[k] = 0.5 * x[k - 3] + slow[k] + rng.normal(scale=0.08)
        seqs.append(x.astype(float))
    return np.array(seqs)


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        rng = np.random.RandomState(seed)
        self.hidden_size = hidden_size
        # weights
        self.Wx = rng.normal(scale=0.1, size=(hidden_size, input_size))
        self.Wh = rng.normal(scale=0.1, size=(hidden_size, hidden_size))
        self.bh = np.zeros(hidden_size)
        self.Wy = rng.normal(scale=0.1, size=(output_size, hidden_size))
        self.by = np.zeros(output_size)

    def forward(self, inputs):
        # inputs: (T, input_size)
        T = inputs.shape[0]
        h = np.zeros((T + 1, self.hidden_size))
        y = np.zeros((T, 1))
        for t in range(T):
            xt = np.asarray(inputs[t]).reshape(-1)
            ht = np.tanh(self.Wx @ xt + self.Wh @ h[t] + self.bh)
            h[t + 1] = ht
            y[t] = (self.Wy @ ht + self.by)
        return y, h

    def bptt(self, inputs, targets, lr=1e-2):
        # full-sequence BPTT, mean-squared error loss
        T = inputs.shape[0]
        y, h = self.forward(inputs)
        err = (y.squeeze() - targets)
        loss = 0.5 * np.mean(err ** 2)
        # gradients init
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dh_next = np.zeros(self.hidden_size)
        for t in reversed(range(T)):
            # derivative of MSE mean wrt prediction
            dy = (y[t].squeeze() - targets[t]) * (1.0 / T)
            dWy += np.outer(np.array([dy]), h[t + 1])
            dby += np.array([dy])
            dh = self.Wy.T @ np.array([dy]) + dh_next
            # back through tanh
            dtanh = (1 - h[t + 1] ** 2) * dh
            dbh += dtanh
            dWx += np.outer(dtanh, np.asarray(inputs[t]).reshape(-1))
            dWh += np.outer(dtanh, h[t])
            dh_next = self.Wh.T @ dtanh
        # gradient step (simple SGD)
        for param, grad in [
            (self.Wx, dWx),
            (self.Wh, dWh),
            (self.bh, dbh),
            (self.Wy, dWy),
            (self.by, dby),
        ]:
            param -= lr * grad
        return float(loss)


def run(seed: int = 42) -> Dict:
    # keep everything deterministic
    np.random.seed(seed)
    # 1. Prepare the data (reduced size for speed)
    seqs = generate_sequences(n_seqs=200, seq_len=40, seed=seed)
    # predict next value for each timestep given previous values
    X = []
    Y = []
    for s in seqs:
        X.append(s[:-1].reshape(-1, 1))
        Y.append(s[1:])
    X = np.stack(X).astype(float)  # shape (n_seqs, T-1, 1)
    Y = np.stack(Y).astype(float)  # shape (n_seqs, T-1)
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    # 2. Build the experiment: smaller/faster RNN
    rnn = SimpleRNN(input_size=1, hidden_size=8, output_size=1, seed=seed)
    # training loop: iterate over sequences (online full-sequence updates)
    epochs = 30
    lr = 0.05
    for ep in range(epochs):
        for i in train_idx:
            _ = rnn.bptt(X[i], Y[i], lr=lr)
        if ep == int(epochs * 0.6):
            lr *= 0.5
    # 3. Evaluate: held-out next-step loss vs last-value baseline
    preds = []
    trues = []
    for i in test_idx:
        y_hat, _ = rnn.forward(X[i])
        preds.append(y_hat.squeeze())
        trues.append(Y[i])
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = float(mean_squared_error(trues, preds))
    r2 = float(r2_score(trues, preds))
    prevs = []
    for i in test_idx:
        prevs.append(X[i].squeeze())
    prevs = np.concatenate(prevs)
    baseline_mse = float(mean_squared_error(trues, prevs))
    # 4. Sequence replay: show one held-out sequence predictions vs truth
    seq_i = test_idx[0]
    y_hat_seq, _ = rnn.forward(X[seq_i])
    y_hat_seq = y_hat_seq.squeeze()
    true_seq = Y[seq_i]
    prev_seq = X[seq_i].squeeze()
    lines = ["t\tprev\tpred\ttrue"]
    for t in range(len(true_seq)):
        lines.append(f"{t}\t{prev_seq[t]:.3f}\t{y_hat_seq[t]:.3f}\t{true_seq[t]:.3f}")
        if t >= 19:
            break
    replay = "\n".join(lines)
    return {
        "r2": r2,
        "mse": mse,
        "baseline_mse": baseline_mse,
        "sequence_replay": replay,
        "model": "simple_numpy_rnn",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
