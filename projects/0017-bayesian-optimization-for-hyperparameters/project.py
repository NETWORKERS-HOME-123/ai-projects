import json
from typing import Dict, List, Any
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


def _eval_params(params: Dict[str, Any], X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, seed: int) -> float:
    """Return cross-validated log loss (lower is better)."""
    try:
        # ensure integer and safe ranges
        n_est = max(1, int(params.get("n_estimators", 10)))
        max_d = None
        try:
            max_d = int(params.get("max_depth", 5))
        except Exception:
            max_d = None
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=n_est,
            max_depth=None if max_d <= 0 else max_d,
            random_state=seed,
            n_jobs=1,
        ))
        # use neg_log_loss scoring; take negative to get positive loss
        scores = cross_val_score(clf, X, y, cv=cv, scoring="neg_log_loss", n_jobs=1)
        loss = -float(np.mean(scores))
        return loss
    except Exception:
        return float("nan")


def _sample_candidates(bounds: Dict[str, tuple], n: int, rng: np.random.RandomState) -> List[Dict[str, float]]:
    cand = []
    for _ in range(n):
        c = {}
        for k, (lo, hi) in bounds.items():
            c[k] = rng.uniform(lo, hi)
        cand.append(c)
    return cand


def _to_array(params_list: List[Dict[str, float]]) -> np.ndarray:
    keys = sorted(params_list[0].keys())
    arr = np.array([[p[k] for k in keys] for p in params_list], dtype=float)
    return arr


def run(seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    # 1) Prepare data (smaller to reduce runtime)
    X, y = make_classification(n_samples=400, n_features=10, n_informative=6,
                               n_redundant=0, n_clusters_per_class=1, random_state=seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    # bounded parameter ranges (kept as ints but sampled continuously)
    bounds = {
        "n_estimators": (5, 80),  # integer
        "max_depth": (2, 20),      # integer
    }
    budget = 12  # total evaluations for each method (reduced to avoid long runtimes)

    # 2) Bayesian optimisation setup
    # initial random evaluations
    initial_points = 3
    trials: List[Dict[str, Any]] = []
    params_list = []
    losses = []

    for i in range(initial_points):
        p = {k: rng.uniform(lo, hi) for k, (lo, hi) in bounds.items()}
        loss = _eval_params(p, X, y, cv, seed)
        trials.append({"trial": len(trials), "params": {k: int(round(v)) for k, v in p.items()},
                       "val_loss": loss, "status": "ok" if np.isfinite(loss) else "fail"})
        params_list.append(p)
        losses.append(loss)

    # surrogate: Gaussian process
    kernel = ConstantKernel(1.0) * Matern(length_scale=[10.0, 5.0], nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True)

    # acquisition via Expected Improvement
    def expected_improvement(Xcand: np.ndarray, Xtrain: np.ndarray, ytrain: np.ndarray, gp_model: GaussianProcessRegressor, xi: float = 0.01):
        mu, sigma = gp_model.predict(Xcand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        best = np.nanmin(ytrain)
        imp = best - mu - xi
        Z = imp / sigma
        from scipy.stats import norm
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    # iterative BO loop
    while len(trials) < budget:
        # fit surrogate on finished trials
        Xtrain = _to_array(params_list)
        ytrain = np.array(losses, dtype=float)
        # guard against NaN; only fit on finite
        mask = np.isfinite(ytrain)
        if mask.sum() == 0:
            break
        try:
            gp.fit(Xtrain[mask], ytrain[mask])
        except Exception:
            # fallback: stop BO if GP fails
            break

        # sample candidates and pick by acquisition
        cands = _sample_candidates(bounds, 100, rng)
        Xc = _to_array(cands)
        try:
            eis = expected_improvement(Xc, Xtrain[mask], ytrain[mask], gp)
            best_idx = int(np.argmax(eis))
        except Exception:
            best_idx = rng.randint(len(cands))
        chosen = cands[best_idx]
        loss = _eval_params(chosen, X, y, cv, seed)
        trials.append({"trial": len(trials), "params": {k: int(round(v)) for k, v in chosen.items()},
                       "val_loss": loss, "status": "ok" if np.isfinite(loss) else "fail"})
        params_list.append(chosen)
        losses.append(loss)

    finite_losses = [t["val_loss"] for t in trials if np.isfinite(t["val_loss"])]
    if len(finite_losses) == 0:
        best_bo = float("inf")
        best_params = {}
    else:
        best_bo = min(finite_losses)
        best_params = [t["params"] for t in trials if t["val_loss"] == best_bo][0]

    # 3) Random search baseline: repeated seeds (reduced repeats)
    repeats = 3
    bests_random = []
    random_histories = []
    for r in range(repeats):
        rr = np.random.RandomState(seed + 100 + r)
        best_r = float("inf")
        hist = []
        for i in range(budget):
            p = {k: rr.uniform(lo, hi) for k, (lo, hi) in bounds.items()}
            loss = _eval_params(p, X, y, cv, seed + r)
            hist.append({"trial": i, "params": {k: int(round(v)) for k, v in p.items()},
                         "val_loss": loss, "status": "ok" if np.isfinite(loss) else "fail"})
            if np.isfinite(loss) and loss < best_r:
                best_r = loss
        bests_random.append(best_r)
        random_histories.append(hist)
    avg_best_random = float(np.mean(bests_random))

    # 4) Prepare output: trial history data (suitable for plotting externally)
    output = {
        "best_val_loss": float(best_bo),
        "best_params": best_params,
        "best_val_loss_random": float(avg_best_random),
        "improvement": float(avg_best_random - best_bo) if np.isfinite(best_bo) else float("nan"),
        "trials": trials,
        "random_histories": random_histories,
        "budget": budget,
        "dataset": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
    }
    return output


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
