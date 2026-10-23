import json
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def _synth_clip(seed: int, freqs: np.ndarray, duration: float = 1.0, sr: int = 22050) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi)
        signal += 0.6 * np.sin(2 * np.pi * f * t + phase)
    # add a small amount of band-limited noise
    noise = rng.normal(0, 0.15, size=t.shape)
    signal += noise
    # apply a simple envelope to emulate note decay
    env = 0.8 + 0.2 * np.exp(-3 * t)
    signal *= env
    # normalise
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    return signal


def _spectral_features(x: np.ndarray, sr: int = 22050) -> np.ndarray:
    # Zero crossing rate
    zc = ((x[:-1] * x[1:]) < 0).sum() / len(x)
    # RMS energy
    rms = np.sqrt(np.mean(x * x))
    # FFT based spectral features
    X = np.fft.rfft(x)
    mags = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total = mags.sum() + 1e-12
    centroid = (mags * freqs).sum() / total
    # spectral bandwidth (second central moment)
    bw = np.sqrt(((mags * (freqs - centroid) ** 2).sum()) / total)
    # spectral flatness (geometric mean / arithmetic mean)
    gm = np.exp(np.mean(np.log(mags + 1e-12)))
    am = mags.mean() + 1e-12
    flat = gm / am
    return np.array([zc, rms, centroid, bw, flat], dtype=float)


def _build_dataset(seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    sr = 22050
    genres = ["A", "B", "C"]  # coarse synthetic "genres"
    artists_per_genre = 8
    clips_per_artist = 8
    rows = []
    clip_id = 0
    for gi, g in enumerate(genres):
        for artist_idx in range(artists_per_genre):
            artist_id = f"g{gi}_a{artist_idx}"
            # choose base frequencies to characterise genre
            if g == "A":
                base = np.array([220, 440.0])
            elif g == "B":
                base = np.array([330, 660.0])
            else:
                base = np.array([176, 352.0])
            for k in range(clips_per_artist):
                # vary mixtures and slight pitch jitter and noise
                freqs = base * rng.uniform(0.95, 1.05, size=base.shape)
                # sometimes add a higher harmonic
                if rng.rand() < 0.3:
                    freqs = np.append(freqs, base.mean() * rng.uniform(2.0, 3.0))
                sig = _synth_clip(seed + clip_id, freqs, duration=1.0, sr=sr)
                feats = _spectral_features(sig, sr=sr)
                rows.append({
                    "clip_id": clip_id,
                    "artist": artist_id,
                    "genre": g,
                    "zc": float(feats[0]),
                    "rms": float(feats[1]),
                    "centroid": float(feats[2]),
                    "bandwidth": float(feats[3]),
                    "flatness": float(feats[4]),
                })
                clip_id += 1
    df = pd.DataFrame(rows)
    X = df[["zc", "rms", "centroid", "bandwidth", "flatness"]].values
    y = df["genre"].values
    return df, X, y


def run(seed: int = 42) -> Dict:
    # Build dataset and meta
    df, X, y = _build_dataset(seed=seed)
    rng = np.random.RandomState(seed)
    # Split by artist: ensure no artist in both train and test
    artists = df[["artist", "genre"]].drop_duplicates().reset_index(drop=True)
    train_artists, test_artists = train_test_split(
        artists,
        test_size=0.25,
        stratify=artists["genre"],
        random_state=seed,
    )
    train_mask = df["artist"].isin(train_artists["artist"])
    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_test = X[~train_mask.values]
    y_test = y[~train_mask.values]

    clf = RandomForestClassifier(n_estimators=150, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    acc = float(accuracy_score(y_test, y_pred))

    out = {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "model": "RandomForestClassifier",
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
