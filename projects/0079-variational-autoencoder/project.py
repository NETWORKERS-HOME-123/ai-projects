import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score


def run(seed: int = 42) -> Dict:
    """Run a small PCA-based VAE-like pipeline on sklearn digits.

    Returns a JSON-serialisable dict with reconstruction and KL metrics.
    """
    rng = np.random.RandomState(seed)

    digits = load_digits()
    X = digits.data.astype(float)  # shape (n_samples, 64)

    # Normalise to a stable numeric scale (keep values as floats)
    X = X / X.max()

    # Train / test split
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=seed)

    # Encoder/decoder: PCA will serve as a linear encoder producing latent means
    n_components = 16
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_train)

    # Encoder outputs: mu (PCA coefficients)
    mu_train = pca.transform(X_train)
    mu_test = pca.transform(X_test)

    # For illustration compute a per-dimension log-variance from the train latent spread
    # This mimics a learnt logvar but is estimated analytically here.
    eps = 1e-6
    latent_var = np.var(mu_train, axis=0) + eps
    logvar = np.log(latent_var)

    # Reparameterised sampling (one sample per input)
    std = np.sqrt(latent_var)
    # Expand to match test samples
    z_test = mu_test + rng.normal(size=mu_test.shape) * std[np.newaxis, :]

    # Decoder: inverse transform from PCA gives reconstructed inputs
    X_recon = pca.inverse_transform(z_test)

    # Clip reconstructions to original data domain [0,1]
    X_recon = np.clip(X_recon, 0.0, 1.0)

    # Reconstruction metric (MSE) on held-out set
    recon_mse = mean_squared_error(X_test, X_recon)

    # KL divergence between q(z|x)=N(mu, diag(var)) and p(z)=N(0,I)
    # KL = 0.5 * sum(var + mu^2 - 1 - logvar) per sample; here var/logvar are shared across samples
    # compute per-sample KL using the sample's mu and the shared var
    var = latent_var
    mu_sq = mu_test ** 2
    # sum over latent dims
    kl_per_sample = 0.5 * (np.sum(var[np.newaxis, :] + mu_sq - 1.0 - logvar[np.newaxis, :], axis=1))
    kl_mean = float(np.mean(kl_per_sample))

    # Evidence lower bound (ELBO) estimate (negative loss) for reporting convenience
    # We treat MSE as the reconstruction term (scaled) and add KL.
    elbo = - (float(recon_mse) + kl_mean)

    # Explained variance between original and reconstruction (1.0 is perfect)
    expl_var = float(explained_variance_score(X_test, X_recon, multioutput='uniform_average'))

    results = {
        "explained_variance": expl_var,
        "reconstruction_mse": float(recon_mse),
        "kl_mean": kl_mean,
        "elbo": float(elbo),
        "latent_dim": int(n_components)
    }
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
