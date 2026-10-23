import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # 1. Prepare data: standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=seed
    )

    # 2. Build experiment: PCA and Isomap (both to 2D)
    pca = PCA(n_components=2, random_state=seed)
    Xp_train = pca.fit_transform(X_train)
    Xp_test = pca.transform(X_test)

    isomap = Isomap(n_components=2, n_neighbors=5)
    Xi_train = isomap.fit_transform(X_train)
    Xi_test = isomap.transform(X_test)

    # 3. Evaluate: reconstruction for PCA (MSE in the scaled feature space)
    Xp_test_recon = pca.inverse_transform(Xp_test)
    pca_recon_mse = float(np.mean((Xp_test_recon - X_test) ** 2))

    # neighbourhood preservation: k-NN overlap between original and embedding (train neighbourhoods)
    # compute neighbours in training sets for original and embeddings
    def knn_indices(matrix, k):
        # compute pairwise euclidean distance
        d = np.sum((matrix[:, None, :] - matrix[None, :, :]) ** 2, axis=2)
        # exclude self by setting large distance on diagonal
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1)[:, :k]

    k = 5
    knn_train_orig = knn_indices(X_train, k)
    knn_train_pca = knn_indices(Xp_train, k)
    knn_train_iso = knn_indices(Xi_train, k)

    # For each test point, find its nearest neighbours among training points in original and embedding spaces
    def knn_overlap_for_tests(Xtr, Xte, knn_tr_orig, knn_tr_emb, k):
        # find indices of k nearest neighbours in training set for each test point
        d_orig = np.sum((Xtr[None, :, :] - Xte[:, None, :]) ** 2, axis=2)
        idx_orig = np.argsort(d_orig, axis=1)[:, :k]
        d_emb = np.sum((knn_tr_emb.dtype == object and Xtr is None) * 0, axis=0)  # dummy to keep style
        # but actually compute for embedding space using matching shapes supplied
        return _knn_overlap_generic(Xtr, Xte, knn_tr_orig=None, k=k)

    # Simpler approach: compute neighbours in training set and neighbours of each test point w.r.t training set
    def knn_overlap(Xtrain, Xtrain_emb, Xtest, Xtrain_emb_flag=False):
        # neighbors in original training space for training points already computed
        d_test_orig = np.sum((Xtrain[None, :, :] - Xtest[:, None, :]) ** 2, axis=2)
        idx_test_orig = np.argsort(d_test_orig, axis=1)[:, :k]
        d_test_emb = np.sum((Xtrain_emb[None, :, :] - (Xp_test if Xtrain_emb_flag else Xp_test)[:, None, :]) ** 2, axis=2)
        # The above line is awkward; instead compute embedding distances using the provided embedding of train and test
        d_test_emb = np.sum((Xtrain_emb[None, :, :] - (Xi_test if np.array_equal(Xtrain_emb, Xi_train) else Xp_test)[:, None, :]) ** 2, axis=2)
        # To avoid this confusion, compute separately for PCA and Isomap below
        return None

    # To be clear and deterministic: implement overlap for PCA and Isomap explicitly
    def knn_overlap_pair(Xtr_orig, Xtr_emb, Xte_orig, Xte_emb, k):
        # neighbours of each test point among training points in original space
        d_to_tr_orig = np.sum((Xtr_orig[None, :, :] - Xte_orig[:, None, :]) ** 2, axis=2)
        idx_to_tr_orig = np.argsort(d_to_tr_orig, axis=1)[:, :k]
        d_to_tr_emb = np.sum((Xtr_emb[None, :, :] - Xte_emb[:, None, :]) ** 2, axis=2)
        idx_to_tr_emb = np.argsort(d_to_tr_emb, axis=1)[:, :k]
        # compute overlap per test point
        overlaps = []
        for a, b in zip(idx_to_tr_orig, idx_to_tr_emb):
            set_a = set(a.tolist())
            set_b = set(b.tolist())
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            overlaps.append(inter / union if union > 0 else 0.0)
        return float(np.mean(overlaps))

    knn_overlap_pca = knn_overlap_pair(X_train, Xp_train, X_test, Xp_test, k)
    knn_overlap_isomap = knn_overlap_pair(X_train, Xi_train, X_test, Xi_test, k)

    # 4. Downstream held-out quality: simple LogisticRegression on original features and on 2D embeddings
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    acc_original = float(accuracy_score(y_test, clf.predict(X_test)))

    clf_pca = LogisticRegression(max_iter=1000, random_state=seed)
    clf_pca.fit(Xp_train, y_train)
    acc_pca2d = float(accuracy_score(y_test, clf_pca.predict(Xp_test)))

    clf_iso = LogisticRegression(max_iter=1000, random_state=seed)
    clf_iso.fit(Xi_train, y_train)
    acc_isomap2d = float(accuracy_score(y_test, clf_iso.predict(Xi_test)))

    # Construct simple textual explanations (short)
    explanations = {
        "pca_explains": "PCA is a linear projection that preserves global variance; inverse_transform allows reconstruction.",
        "isomap_explains": "Isomap is a nonlinear embedding aiming to preserve local geodesic structure; no exact inverse is provided."
    }

    return {
        "pca_recon_mse": pca_recon_mse,
        "knn_overlap_pca": knn_overlap_pca,
        "knn_overlap_isomap": knn_overlap_isomap,
        "acc_original": acc_original,
        "acc_pca2d": acc_pca2d,
        "acc_isomap2d": acc_isomap2d,
        "explanations": explanations
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
