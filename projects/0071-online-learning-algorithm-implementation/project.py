import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def make_stream(n_samples: int = 1000, n_features: int = 10, change_at: int = 500, seed: int = 42):
    rng = np.random.RandomState(seed)
    # Phase A: class signal introduced on half the features
    half = n_features // 2
    Xa = rng.normal(loc=0.0, scale=1.0, size=(change_at, n_features))
    # weak structure: alternate sign on first half
    Xa[:, :half] += (np.arange(change_at) % 2)[:, None] * -1.0
    # make labels depend on a simple linear signal (learnable), with small noise
    ya = (Xa[:, :half].sum(axis=1) + rng.normal(scale=0.5, size=change_at) > 0).astype(int)
    # Phase B: introduce a shift to create drift: means move by +2 on first half
    Xb = rng.normal(loc=0.0, scale=1.0, size=(n_samples - change_at, n_features))
    Xb[:, :half] += (np.arange(n_samples - change_at) % 2)[:, None] * 1.0 + 2.0
    # labels still follow the same linear rule but reflect the shifted features
    yb = (Xb[:, :half].sum(axis=1) + rng.normal(scale=0.5, size=n_samples - change_at) > 0).astype(int)
    X = np.vstack([Xa, Xb])
    y = np.concatenate([ya, yb])
    # time ordering preserved
    return X, y


def run(seed: int = 42) -> Dict[str, Any]:
    # Configuration
    n_samples = 1000
    n_features = 10
    change_at = 500
    batch_size = 20
    reset_threshold = 0.15  # drop in rolling accuracy that triggers reset
    rolling_window = 10  # number of batches used for rolling accuracy

    X, y = make_stream(n_samples=n_samples, n_features=n_features, change_at=change_at, seed=seed)

    scaler = StandardScaler()
    # Fit scaler on first small window to ensure consistent feature handling
    init_window = 100
    scaler.fit(X[:init_window])

    # Prepare classifiers
    online = SGDClassifier(loss='log_loss', max_iter=1, tol=None, learning_rate='constant', eta0=0.01, random_state=seed)
    # We need to call partial_fit with classes first
    classes = np.unique(y)
    # Initialize online on first batch
    start_idx = 0
    end_idx = batch_size
    X0 = scaler.transform(X[start_idx:end_idx])
    y0 = y[start_idx:end_idx]
    online.partial_fit(X0, y0, classes=classes)

    # Frozen baseline trained once on initial window and never updated
    frozen = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=seed)
    frozen.fit(scaler.transform(X[:init_window]), y[:init_window])

    # Prequential loop
    n_batches = int(np.ceil(n_samples / batch_size))
    online_preds = []
    frozen_preds = []
    online_trues = []

    rolling_online_acc = []
    drift_flags = []
    resets = 0

    # helper to compute batch accuracy
    def acc(p, t):
        return float((p == t).mean())

    for b in range(n_batches):
        i0 = b * batch_size
        i1 = min((b + 1) * batch_size, n_samples)
        Xb = scaler.transform(X[i0:i1])
        yb = y[i0:i1]
        # Predict with current online and with frozen baseline (prequential: predict before update)
        p_online = online.predict(Xb)
        p_frozen = frozen.predict(Xb)
        online_preds.append(acc(p_online, yb))
        frozen_preds.append(acc(p_frozen, yb))
        online_trues.extend(yb.tolist())

        # Update rolling accuracy and check drift
        rolling_online_acc.append(online_preds[-1])
        if len(rolling_online_acc) > rolling_window:
            rolling_online_acc.pop(0)
        if len(rolling_online_acc) == rolling_window:
            # compare recent mean to earlier mean (naive drift indicator)
            recent = np.mean(rolling_online_acc)
            # compute baseline reference: mean of first rolling_window batches
            ref_end = rolling_window
            ref_start = 0
            # make a simple reference from the earliest available batch accuracies
            ref_values = online_preds[:ref_end]
            ref_mean = np.mean(ref_values) if ref_values else recent
            if ref_mean - recent > reset_threshold:
                drift_flags.append(1)
                # Reset control: reinitialise online model to start-over state
                online = SGDClassifier(loss='log_loss', max_iter=1, tol=None, learning_rate='constant', eta0=0.01, random_state=seed)
                # Re-fit on the most recent initial window to avoid cold start
                warm_idx_end = min(init_window, i1)
                online.partial_fit(scaler.transform(X[:warm_idx_end]), y[:warm_idx_end], classes=classes)
                resets += 1
            else:
                drift_flags.append(0)
        else:
            drift_flags.append(0)

        # Finally update the online model with the current batch (prequential update)
        online.partial_fit(Xb, yb)

    # Summarise
    mean_online = float(np.mean(online_preds))
    mean_frozen = float(np.mean(frozen_preds))
    drift_fraction = float(np.mean(drift_flags))

    result = {
        'online_mean_accuracy': mean_online,
        'frozen_mean_accuracy': mean_frozen,
        'drift_fraction': drift_fraction,
        'resets': int(resets),
        'n_batches': int(n_batches)
    }
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
