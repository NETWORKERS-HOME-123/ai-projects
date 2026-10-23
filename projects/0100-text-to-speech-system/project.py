import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def normalize_text(s: str) -> str:
    """A small deterministic text normaliser example.
    - lowercases
    - expands a couple of common abbreviations deterministically
    - spells digits as words (0-9)
    """
    s = s.lower()
    abbr = {"mr.": "mister", "mrs.": "misses", "dr.": "doctor", "st.": "street"}
    for k, v in abbr.items():
        s = s.replace(k, v)
    digits = {str(i): ["zero","one","two","three","four","five","six","seven","eight","nine"][i] for i in range(10)}
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(digits[ch])
        else:
            out.append(ch)
    # collapse multiple spaces
    norm = "".join(out).replace("  ", " ").strip()
    return norm


def toy_synth_waveform(text: str, sr: int = 16000, duration_s: float = 0.5) -> np.ndarray:
    """Deterministic toy synthesiser that maps text to a short waveform.

    This is intentionally simple: it combines up to three sine components whose
    frequencies and amplitudes are derived from hash values of the normalised text.
    It is not a real TTS model and does not produce a human voice.
    """
    norm = normalize_text(text)
    # deterministic numeric seeds from text
    h = abs(hash(norm))
    rng = np.random.RandomState(h % (2**32))
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # derive 3 frequencies in speech-like range but simple
    base = 120 + (h % 200)  # between 120 and 319
    freqs = base + np.array([0, 200, 400]) + (rng.randint(0, 50, size=3))
    amps = 0.5 * (1.0 / (1.0 + np.arange(3))) * (1.0 + (rng.rand(3) - 0.5) * 0.2)
    phase = rng.rand(3) * 2 * np.pi
    wave = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, phase):
        wave += a * np.sin(2 * np.pi * f * t + p)
    # apply a deterministic envelope to avoid clicks
    env = np.minimum(1.0, (t * 5.0)) * np.minimum(1.0, (duration_s - t) * 5.0)
    wave *= env
    # normalise
    maxabs = np.max(np.abs(wave))
    if maxabs > 0:
        wave = wave / maxabs * 0.9
    return wave


def text_features(phrases: List[str]) -> pd.DataFrame:
    rows = []
    for s in phrases:
        norm = normalize_text(s)
        tokens = norm.split()
        rows.append({
            "chars": len(norm),
            "tokens": len(tokens),
            "digits": sum(ch.isdigit() for ch in s),
            "abbr_count": int(any(p in s for p in ["mr.", "mrs.", "dr.", "st."])),
            "uppercase_original": sum(1 for ch in s if ch.isupper()),
        })
    return pd.DataFrame(rows)


def run(seed: int = 42) -> Dict[str, float]:
    """Builds synthetic data, trains a regression model and returns metrics.

    Returns a JSON-serialisable dict with keys:
    - r2_train, r2_test, mae_test, example_wave_mean
    """
    # 1) Create many synthetic "text-like" examples
    rng = np.random.RandomState(seed)
    n_samples = 400
    # We'll use make_regression to produce a continuous target analogous to a quality score.
    X_reg, y = make_regression(n_samples=n_samples, n_features=8, noise=15.0, random_state=seed)

    # Add deterministic text-derived features using a small pool of phrases
    pool = [
        "Call Dr. Kumar at 555-0101",
        "Meet Mr. Singh at 5pm on St. Martin's Avenue",
        "Temperature is 30C",
        "She scored 98 out of 100",
        "Update: flight at 07:45",
    ]
    phrases = [pool[i % len(pool)] for i in range(n_samples)]
    df_text = text_features(phrases)

    # combine into final feature matrix
    X = np.hstack([X_reg, df_text.values])

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)

    # 3) Fit a simple, deterministic model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4) Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_train = float(r2_score(y_train, y_pred_train))
    r2_test = float(r2_score(y_test, y_pred_test))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))

    # 5) Produce an example normalisation + toy waveform summary
    example_phrase = "Dr. A. B. Patel, meeting at 9am"
    wave = toy_synth_waveform(example_phrase, sr=8000, duration_s=0.4)
    example_wave_mean = float(np.mean(np.abs(wave)))

    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_test": mae_test,
        "example_wave_mean": example_wave_mean,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
