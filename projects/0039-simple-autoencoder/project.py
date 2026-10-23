import json
from typing import Dict, Any, List
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score


def run(seed: int = 42) -> Dict[str, Any]:
    # 1. Load data
    data = load_digits()
    X = data.data.astype(float)  # shape (n_samples, 64)

    # 2. Train/test split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)

    # 3. Scale using training statistics only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Build a small MLP autoencoder: symmetric with a narrow bottleneck
    input_dim = X_train_s.shape[1]
    latent_dim = 8  # constrained latent dimension for the bottleneck
    # hidden layers: encoder (32 -> latent_dim) and decoder (32)
    hidden_layers = (32, latent_dim, 32)

    # Use MLPRegressor to reconstruct inputs
    # use a solver and settings that are deterministic with random_state and stable here
    model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                         activation='relu',
                         solver='adam',
                         max_iter=1000,
                         tol=1e-6,
                         learning_rate_init=0.001,
                         random_state=seed)

    model.fit(X_train_s, X_train_s)

    # 5. Evaluate reconstruction on held-out set
    X_test_rec_s = model.predict(X_test_s)
    test_mse = float(mean_squared_error(X_test_s, X_test_rec_s))
    # use variance_weighted multioutput R2 to be stable across outputs
    test_r2 = float(r2_score(X_test_s, X_test_rec_s, multioutput='variance_weighted'))

    # 6. PCA baseline with same latent_dim (fit on train then transform+inverse_transform)
    pca = PCA(n_components=latent_dim, random_state=seed)
    pca.fit(X_train_s)
    X_test_pca_rec_s = pca.inverse_transform(pca.transform(X_test_s))
    pca_mse = float(mean_squared_error(X_test_s, X_test_pca_rec_s))

    # 7. Find examples with largest reconstruction error (on scaled space)
    per_sample_mse = np.mean((X_test_s - X_test_rec_s) ** 2, axis=1)
    top_k = 5
    worst_idx = np.argsort(per_sample_mse)[-top_k:][::-1]

    examples: List[Dict[str, Any]] = []
    for i in worst_idx:
        examples.append({
            "test_index": int(i),
            "mse": float(per_sample_mse[i]),
            # return original and reconstructed on the unscaled pixel range for inspectability
            "original": X_test[i].tolist(),
            "reconstruction": scaler.inverse_transform(X_test_rec_s[i].reshape(1, -1)).ravel().tolist(),
        })

    out = {
        "dataset": "digits",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "latent_dim": int(latent_dim),
        "test_mse_scaled": test_mse,
        "pca_mse_scaled": pca_mse,
        "reconstruction_r2": test_r2,
        "top_errors": examples,
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
