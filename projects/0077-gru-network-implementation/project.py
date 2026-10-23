import json, time
from typing import Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _tanh(x):
    return np.tanh(x)

class GRUReservoir:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = rng.normal(scale=0.5, size=(hidden_size, input_size))
        self.W_r = rng.normal(scale=0.5, size=(hidden_size, input_size))
        self.W_h = rng.normal(scale=0.5, size=(hidden_size, input_size))
        self.U_z = rng.normal(scale=0.5, size=(hidden_size, hidden_size))
        self.U_r = rng.normal(scale=0.5, size=(hidden_size, hidden_size))
        self.U_h = rng.normal(scale=0.5, size=(hidden_size, hidden_size))
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x: np.ndarray, reset_hidden: bool = False) -> np.ndarray:
        batch, seq_len, _ = x.shape
        h = np.zeros((batch, self.hidden_size))
        for t in range(seq_len):
            xt = x[:, t, :].T
            hz = _sigmoid(self.W_z @ xt + self.U_z @ h.T + self.b_z)
            hr = _sigmoid(self.W_r @ xt + self.U_r @ h.T + self.b_r)
            h_tilde = _tanh(self.W_h @ xt + self.U_h @ (hr * h.T) + self.b_h)
            h = ((1 - hz) * h.T + hz * h_tilde).T
        return h

class SimpleRNNReservoir:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng = np.random.RandomState(seed + 1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = rng.normal(scale=0.5, size=(hidden_size, input_size))
        self.U = rng.normal(scale=0.5, size=(hidden_size, hidden_size))
        self.b = np.zeros((hidden_size, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = x.shape
        h = np.zeros((batch, self.hidden_size))
        for t in range(seq_len):
            xt = x[:, t, :].T
            h = _tanh(self.W @ xt + self.U @ h.T + self.b).T
        return h

def make_dataset(n_samples: int, seq_len: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, seq_len, 1)).astype(np.float32)
    y = (X[:, 0, 0].astype(int) & X[:, -1, 0].astype(int)).astype(int)
    return X, y

def run(seed: int = 42) -> Dict[str, object]:
    start_time = time.time()
    np.random.seed(seed)
    n_samples = 1000
    seq_short = 8
    seq_long = 20
    input_size = 1
    hidden_size = 16
    Xs, ys = make_dataset(n_samples, seq_short, seed=seed)
    Xl, yl = make_dataset(n_samples, seq_long, seed=seed + 5)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.25, random_state=seed, stratify=ys)
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.25, random_state=seed, stratify=yl)
    gru = GRUReservoir(input_size=input_size, hidden_size=hidden_size, seed=seed)
    rnn = SimpleRNNReservoir(input_size=input_size, hidden_size=hidden_size, seed=seed)
    Zs_train = gru.forward(Xs_train)
    Zs_test = gru.forward(Xs_test)
    R_s_train = rnn.forward(Xs_train)
    R_s_test = rnn.forward(Xs_test)
    Xs_flat_train = Xs_train.reshape(Xs_train.shape[0], -1)
    Xs_flat_test = Xs_test.reshape(Xs_test.shape[0], -1)
    clf_gru = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=200)
    clf_gru.fit(Zs_train, ys_train)
    ypred_gru_prob = clf_gru.predict_proba(Zs_test)[:, 1]
    ypred_gru = clf_gru.predict(Zs_test)
    clf_rnn = LogisticRegression(random_state=seed + 1, solver="lbfgs", max_iter=200)
    clf_rnn.fit(R_s_train, ys_train)
    ypred_rnn = clf_rnn.predict(R_s_test)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed)
    mlp.fit(Xs_flat_train, ys_train)
    ypred_mlp = mlp.predict(Xs_flat_test)
    acc_gru = float(accuracy_score(ys_test, ypred_gru))
    loss_gru = float(log_loss(ys_test, ypred_gru_prob, labels=[0, 1]))
    acc_rnn = float(accuracy_score(ys_test, ypred_rnn))
    acc_mlp = float(accuracy_score(ys_test, ypred_mlp))
    Zl_train = gru.forward(Xl_train)
    Zl_test = gru.forward(Xl_test)
    clf_gru_long = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=200)
    clf_gru_long.fit(Zl_train, yl_train)
    acc_gru_long = float(accuracy_score(yl_test, clf_gru_long.predict(Zl_test)))
    runtime = time.time() - start_time
    demo = {
        "short_seq_shape": list(Xs.shape),
        "long_seq_shape": list(Xl.shape),
        "gru_hidden_shape": [hidden_size],
        "explanation": (
            "The GRUReservoir.forward iterates over timesteps, updating a hidden state h. "
            "Hidden state is initialised to zeros for each sequence (explicit reset). The reservoir "
            "weights are fixed; learning happens in the linear readout (LogisticRegression).")
    }
    result = {
        "gru_test_accuracy": acc_gru,
        "gru_test_loss": loss_gru,
        "rnn_test_accuracy": acc_rnn,
        "mlp_test_accuracy": acc_mlp,
        "gru_longseq_test_accuracy": acc_gru_long,
        "runtime_seconds": float(runtime),
        "sequence_length_short": int(seq_short),
        "sequence_length_long": int(seq_long),
        "demo": demo,
    }
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
