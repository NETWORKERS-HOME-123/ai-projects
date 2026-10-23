import json
from typing import Dict, List, Tuple
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _initialise_grid(shape: Tuple[int, int], dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(shape[0], shape[1], dim))


def _grid_coordinates(shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    coords = np.indices((rows, cols)).transpose(1, 2, 0)
    return coords.reshape(-1, 2)


def _neighbourhood_kernel(bmu_coord: Tuple[int, int], coords: np.ndarray, radius: float) -> np.ndarray:
    # Gaussian kernel based on grid distance
    d2 = np.sum((coords - np.array(bmu_coord)) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * (radius ** 2 + 1e-12)))


def _find_bmus(prototypes: np.ndarray, X: np.ndarray) -> np.ndarray:
    # prototypes: (R, C, D) -> reshape to (R*C, D)
    flat = prototypes.reshape(-1, prototypes.shape[2])
    # compute squared distances
    d = np.sum((X[:, None, :] - flat[None, :, :]) ** 2, axis=2)
    return np.argmin(d, axis=1)  # index into flat


def _topographic_error(flat_protos: np.ndarray, X: np.ndarray, coords: np.ndarray) -> float:
    # For each sample, find first and second BMU indices, check adjacency
    d = np.sum((X[:, None, :] - flat_protos[None, :, :]) ** 2, axis=2)
    order = np.argsort(d, axis=1)
    first = order[:, 0]
    second = order[:, 1]
    first_coord = coords[first]
    second_coord = coords[second]
    manhattan = np.abs(first_coord - second_coord).sum(axis=1)
    # adjacent if manhattan distance == 1 (4-neighbourhood)
    not_adjacent = np.sum(manhattan > 1)
    return float(not_adjacent / X.shape[0])


def run(seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    # 1. Prepare data
    X, _ = make_blobs(n_samples=300, centers=4, n_features=3, cluster_std=1.0, random_state=seed)
    X = StandardScaler().fit_transform(X)
    X_train, X_test = train_test_split(X, test_size=0.33, random_state=seed)

    # SOM hyperparameters
    grid_shape = (6, 6)  # modest grid
    n_epochs = 80
    initial_lr = 0.5
    initial_radius = max(grid_shape) / 2.0

    # initialise prototypes
    prototypes = _initialise_grid(grid_shape, X.shape[1], rng)
    coords = _grid_coordinates(grid_shape)  # (R*C, 2)
    flat_coords = coords

    # training loop
    n_neurons = grid_shape[0] * grid_shape[1]
    flat_protos = prototypes.reshape(n_neurons, X.shape[1])

    for epoch in range(n_epochs):
        # decays
        t = epoch / max(1, n_epochs - 1)
        lr = initial_lr * (1.0 - t)
        radius = initial_radius * (1.0 - t)
        # shuffle training data
        order = rng.permutation(X_train.shape[0])
        for i in order:
            x = X_train[i]
            # find BMU index
            d2 = np.sum((flat_protos - x[None, :]) ** 2, axis=1)
            bmu_idx = int(np.argmin(d2))
            bmu_coord = coords[bmu_idx]
            # neighbourhood kernel
            kernel = _neighbourhood_kernel(bmu_coord, coords, radius)
            # update prototypes (vectorised over all neurons)
            influence = (lr * kernel)[:, None]
            flat_protos += influence * (x[None, :] - flat_protos)

    # reshape back
    prototypes = flat_protos.reshape(grid_shape[0], grid_shape[1], X.shape[1])

    # Evaluation on held-out test set
    flat_protos = prototypes.reshape(-1, X.shape[1])
    bmus = _find_bmus(prototypes, X_test)
    dists = np.linalg.norm(X_test - flat_protos[bmus], axis=1)
    quantisation_error = float(np.mean(dists))
    topo_error = _topographic_error(flat_protos, X_test, flat_coords)

    result = {
        "quantisation_error": quantisation_error,
        "topographic_error": topo_error,
        "grid_shape": [grid_shape[0], grid_shape[1]],
        "prototypes": flat_protos.tolist(),
        "assignments": [int(x) for x in bmus.tolist()],
        "seed": int(seed),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
