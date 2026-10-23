import json
from typing import Dict
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(y):
    return y * (1 - y)

class TinyMLP:
    def __init__(self, n_in, n_hidden, seed=42):
        rng = np.random.RandomState(seed)
        # Heuristic small initialisation
        self.W1 = rng.randn(n_in, n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.randn(n_hidden, 1) * 0.5
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = sigmoid(Z2)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, Yhat, Y):
        # binary cross-entropy, stable
        eps = 1e-9
        Yhat = np.clip(Yhat, eps, 1 - eps)
        loss = - np.mean(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))
        return loss

    def backward(self, cache, Y):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[0]
        dZ2 = (A2 - Y) / m  # derivative of BCE with sigmoid output
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def update(self, grads, lr=0.1):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]


def numeric_grad_check(model: TinyMLP, X, Y, eps=1e-5):
    # Compute analytic grads
    Yhat, cache = model.forward(X)
    grads = model.backward(cache, Y)
    # Numeric check for a small subset of parameters: first 3 weights in W1
    numeric_diffs = []
    for idx in [(0, 0), (0, 1), (1, 0)]:
        i, j = idx
        orig = model.W1[i, j]
        model.W1[i, j] = orig + eps
        plus, _ = model.forward(X)
        lplus = model.compute_loss(plus, Y)
        model.W1[i, j] = orig - eps
        minus, _ = model.forward(X)
        lminus = model.compute_loss(minus, Y)
        model.W1[i, j] = orig
        numeric = (lplus - lminus) / (2 * eps)
        analytic = grads["dW1"][i, j] / X.shape[0] * X.shape[0]  # same scale
        numeric_diffs.append(abs(numeric - analytic))
    return float(np.mean(numeric_diffs))


def run(seed: int = 42) -> Dict[str, float]:
    # 1. Data
    X, y = make_moons(n_samples=200, noise=0.2, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # 2. Model
    model = TinyMLP(n_in=X.shape[1], n_hidden=8, seed=seed)

    # 3. Gradient check on a tiny batch
    gc_X = X_train[:10]
    gc_Y = y_train[:10]
    grad_check_diff = numeric_grad_check(model, gc_X, gc_Y)

    # 4. Training loop (simple SGD with full-batch updates, deterministic)
    epochs = 300
    lr = 0.5
    train_loss = None
    for epoch in range(epochs):
        Yhat, cache = model.forward(X_train)
        train_loss = model.compute_loss(Yhat, y_train)
        grads = model.backward(cache, y_train)
        model.update(grads, lr=lr)
        # simple learning rate schedule
        if epoch == 150:
            lr *= 0.3

    # 5. Evaluate
    Yhat_train, _ = model.forward(X_train)
    Yhat_test, _ = model.forward(X_test)
    train_loss = model.compute_loss(Yhat_train, y_train)
    test_loss = model.compute_loss(Yhat_test, y_test)
    test_preds = (Yhat_test >= 0.5).astype(int)
    test_accuracy = float(np.mean(test_preds == y_test))

    # 6. Baseline: logistic regression
    baseline = LogisticRegression(random_state=seed, solver="lbfgs")
    baseline.fit(X_train, y_train.ravel())
    baseline_acc = float(baseline.score(X_test, y_test.ravel()))

    return {
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "baseline_accuracy": float(baseline_acc),
        "grad_check_diff": float(grad_check_diff)
    }

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
