import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def synth_signals(n_speakers=5, sessions_per_speaker=3, samples_per_session=8, sr=8000, duration=1.0, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = n_speakers * sessions_per_speaker * samples_per_session
    X = np.zeros((n_samples, int(sr * duration)))
    labels = []
    sessions = []
    meta = []
    idx = 0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    for sp in range(n_speakers):
        # each speaker has a base harmonic set
        base_freq = 120 + sp * 30  # different speaker fundamental
        harmonics = [1.0, 0.6 + rng.rand() * 0.3, 0.3 + rng.rand() * 0.3]
        for sess in range(sessions_per_speaker):
            # session-specific additive noise / mic colour
            session_noise = 0.01 + rng.rand() * 0.03
            session_shift = rng.randn() * 2.0  # small frequency perturb
            for s in range(samples_per_session):
                # sample-level variability
                amp = 0.6 + rng.rand() * 0.8
                phases = rng.rand(len(harmonics)) * 2 * np.pi
                sig = np.zeros_like(t)
                for h, ph in enumerate(phases):
                    freq = base_freq * (h + 1) + session_shift + rng.randn() * 0.5
                    sig += harmonics[h] * np.sin(2 * np.pi * freq * t + ph)
                # apply amplitude, clip and add noise
                sig = amp * sig
                sig += rng.normal(scale=session_noise, size=sig.shape)
                # lightweight microphone-like filtering: a gentle low-pass via simple convolution
                kernel = np.exp(-np.linspace(0, 1, 9) * (0.5 + rng.rand()))
                kernel = kernel / kernel.sum()
                sig = np.convolve(sig, kernel, mode="same")
                X[idx] = sig.astype(np.float32)
                labels.append(f"sp{sp}")
                sessions.append(f"sess{sess}")
                meta.append({"speaker": sp, "session": sess})
                idx += 1
    return X, np.array(labels), np.array(sessions), pd.DataFrame(meta)


def extract_features(waveforms, sr=8000):
    # Compute simple spectral features per frame: spectral centroid, bandwidth, band energies, zcr
    feats = []
    for sig in waveforms:
        N = len(sig)
        # zero-crossing rate
        zcr = ((np.diff(np.sign(sig)) != 0).sum()) / N
        # FFT
        S = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(N, 1.0 / sr)
        total_energy = S.sum() + 1e-12
        centroid = (freqs * S).sum() / total_energy
        bw = np.sqrt(((freqs - centroid) ** 2 * S).sum() / total_energy)
        # band energies in low/mid/high bands
        e_low = S[(freqs >= 0) & (freqs < 500)].sum() / total_energy
        e_mid = S[(freqs >= 500) & (freqs < 2000)].sum() / total_energy
        e_high = S[(freqs >= 2000)].sum() / total_energy
        feats.append([zcr, centroid, bw, e_low, e_mid, e_high, total_energy])
    return np.array(feats, dtype=float)


def run(seed: int = 42) -> Dict[str, Any]:
    # 1) Prepare synthetic dataset mimicking multiple speakers and sessions
    X_raw, labels, sessions, meta = synth_signals(seed=seed)

    # 2) Extract features
    X_feats = extract_features(X_raw)

    # 3) Split with session-aware holdout: ensure some sessions are unseen per speaker
    # For simplicity, hold out one session index across all speakers
    rng = np.random.RandomState(seed)
    holdout_session = int(rng.randint(0, 3))  # session index to hold out
    is_holdout = (meta["session"] == holdout_session)
    X_train = X_feats[~is_holdout]
    y_train = labels[~is_holdout]
    X_test = X_feats[is_holdout]
    y_test = labels[is_holdout]

    # 4) Standardise
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5) Train a baseline classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train_s, y_train)

    # 6) Predict with an explicit rejection policy: if max_prob < threshold -> unknown
    probs = clf.predict_proba(X_test_s)
    maxp = probs.max(axis=1)
    preds = clf.classes_[probs.argmax(axis=1)]
    # choose threshold conservatively: median of train max probs on validation-like split
    train_probs = clf.predict_proba(X_train_s)
    train_max = train_probs.max(axis=1)
    threshold = max(0.40, float(np.median(train_max) * 0.9))

    preds_with_reject = np.where(maxp >= threshold, preds, "unknown")

    # 7) Metrics
    overall_acc = accuracy_score(y_test, preds_with_reject)
    # Known-speaker accuracy (ignoring unknown predictions as rejects): only consider non-unknown preds
    known_mask = preds_with_reject != "unknown"
    if known_mask.sum() > 0:
        known_acc = accuracy_score(y_test[known_mask], preds_with_reject[known_mask])
    else:
        known_acc = 0.0
    # False accept rate: fraction of unknowns accepted as a wrong known speaker
    # In closed-set synthetic data there are no true unknown speakers; to measure false accepts we
    # simulate unknown trials by taking a small set of artificially shifted signals
    unk_X, unk_labels, _, _ = synth_signals(n_speakers=2, sessions_per_speaker=1, samples_per_session=6, seed=seed + 7)
    unk_feats = extract_features(unk_X)
    unk_feats_s = scaler.transform(unk_feats)
    unk_probs = clf.predict_proba(unk_feats_s)
    unk_maxp = unk_probs.max(axis=1)
    unk_preds = clf.classes_[unk_probs.argmax(axis=1)]
    unk_accepted = (unk_maxp >= threshold)
    false_accept_rate = float(unk_accepted.sum() / len(unk_accepted))

    # Unknown reject rate on held-out: fraction of test samples labelled unknown
    unknown_reject_rate = float((preds_with_reject == "unknown").sum() / len(preds_with_reject))

    result = {
        "overall_accuracy": float(overall_acc),
        "known_speaker_accuracy": float(known_acc),
        "unknown_reject_rate": float(unknown_reject_rate),
        "false_accept_rate": float(false_accept_rate),
        "threshold": float(threshold),
        "holdout_session": int(holdout_session),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
