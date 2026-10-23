import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _seeded_init(shape, seed):
    rng = np.random.RandomState(seed)
    return rng.normal(scale=0.1, size=shape)


class SimpleAutoencoder:
    """Single-hidden-bottleneck autoencoder implemented with NumPy."""

    def __init__(self, input_dim: int, latent_dim: int, seed: int = 0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seed = seed
        # Encoder weights and bias
        self.W_enc = _seeded_init((latent_dim, input_dim), seed + 1)
        self.b_enc = np.zeros((latent_dim,))
        # Decoder weights and bias
        self.W_dec = _seeded_init((input_dim, latent_dim), seed + 2)
        self.b_dec = np.zeros((input_dim,))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _sigmoid_grad(s):
        return s * (1 - s)

    def encode(self, X: np.ndarray) -> np.ndarray:
        z = X.dot(self.W_enc.T) + self.b_enc
        return self._sigmoid(z)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        x = Z.dot(self.W_dec.T) + self.b_dec
        return self._sigmoid(x)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(X))

    def train(self, X_train: np.ndarray, X_val: np.ndarray, *, epochs=200, lr=0.5, verbose=False):
        # Simple SGD with batch gradient over full dataset (batch GD) for stability
        for epoch in range(epochs):
            # Forward
            Z = self.encode(X_train)
            X_hat = self.decode(Z)
            # Reconstruction error (MSE) and gradient
            err = X_hat - X_train  # shape (n, input_dim)
            # Compute gradients for decoder
            grad_dec_pre = err * self._sigmoid_grad(X_hat)
            dW_dec = grad_dec_pre.T.dot(Z) / X_train.shape[0]
            db_dec = grad_dec_pre.mean(axis=0)
            # Backprop to latent
            grad_latent = grad_dec_pre.dot(self.W_dec) * self._sigmoid_grad(Z)
            dW_enc = grad_latent.T.dot(X_train) / X_train.shape[0]
            db_enc = grad_latent.mean(axis=0)
            # Update
            self.W_dec -= lr * dW_dec
            self.b_dec -= lr * db_dec
            self.W_enc -= lr * dW_enc
            self.b_enc -= lr * db_enc
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                mse = float(((X_hat - X_train) ** 2).mean())
                print(f"epoch {epoch} mse {mse:.6f}")


def _prepare_data(seed: int = 0):
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.copy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed + 1, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def run(seed: int = 42) -> Dict:
    """Run the experiment. Returns a JSON-serialisable dict of metrics.

    The main reported metric is 'ae_clf_acc_8' (autoencoder downstream accuracy with latent_dim=8).
    """
    np.random.seed(seed)
    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_data(seed=seed)

    input_dim = X_train.shape[1]
    latent_dims = [2, 8, 16]
    results: List[Dict] = []

    for ld in latent_dims:
        ae = SimpleAutoencoder(input_dim=input_dim, latent_dim=ld, seed=seed)
        # Train with the combined train+val for stability but keep splits deterministic
        X_combined = np.vstack([X_train, X_val])
        ae.train(X_combined, X_val, epochs=300, lr=1.0, verbose=False)
        # Encode test
        Z_test = ae.encode(X_test)
        X_rec_ae = ae.reconstruct(X_test)
        ae_mse = float(((X_rec_ae - X_test) ** 2).mean())
        # PCA baseline
        pca = PCA(n_components=ld, random_state=seed)
        pca.fit(X_train)
        Z_pca_test = pca.transform(X_test)
        X_rec_pca = pca.inverse_transform(Z_pca_test)
        pca_mse = float(((X_rec_pca - X_test) ** 2).mean())
        # Downstream classification on latent features (LogisticRegression)
        # Use a solver that supports multiclass classification
        # older sklearn versions may not accept multi_class kwarg, omit it for compatibility
        clf_ae = LogisticRegression(max_iter=500, solver='lbfgs', random_state=seed)
        clf_pca = LogisticRegression(max_iter=500, solver='lbfgs', random_state=seed)
        # train classifiers on train split encoded by each method
        Z_train_ae = ae.encode(X_train)
        Z_train_pca = pca.transform(X_train)
        clf_ae.fit(Z_train_ae, y_train)
        clf_pca.fit(Z_train_pca, y_train)
        ae_acc = float(clf_ae.score(Z_test, y_test))
        pca_acc = float(clf_pca.score(Z_pca_test, y_test))
        results.append({
            'latent_dim': int(ld),
            'ae_mse': ae_mse,
            'pca_mse': pca_mse,
            'ae_clf_acc': ae_acc,
            'pca_clf_acc': pca_acc,
            # small example of information loss: mean absolute difference between original and AE rec for first test sample
            'example_loss_l1_first_sample': float(abs(X_test[0] - X_rec_ae[0]).mean()),
        })

    # Build a small summary and pick the main metric for latent_dim=8
    summary = {f"ld_{r['latent_dim']}": r for r in results}
    # main metric
    ae_clf_acc_8 = next(r['ae_clf_acc'] for r in results if r['latent_dim'] == 8)
    out = {
        'dataset': 'digits_builtin',
        'model': 'simple_numpy_autoencoder',
        'latent_dims': [int(x) for x in latent_dims],
        'results': summary,
        'ae_clf_acc_8': float(ae_clf_acc_8),
    }
    return out


if __name__ == "__main__":
    import sys

    print(json.dumps(run(), indent=2))
