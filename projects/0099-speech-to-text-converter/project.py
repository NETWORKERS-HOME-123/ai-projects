import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def synth_clip(phrase_id, vocab_wave_map, sample_rate=8000, duration=0.8, noise_level=0.0, stretch=1.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration * stretch, int(sample_rate * duration * stretch), endpoint=False)
    signal = np.zeros_like(t)
    for f, amp in vocab_wave_map[phrase_id]:
        signal += amp * np.sin(2 * np.pi * f * t)
    envelope = np.linspace(0.1, 1.0, signal.size)
    signal *= envelope
    if noise_level > 0:
        signal += noise_level * rng.randn(*signal.shape)
    signal = signal / (np.abs(signal).max() + 1e-9)
    return signal, sample_rate


def extract_band_features(signal, sample_rate, n_bands=6, top_k=4):
    x = signal - np.mean(signal)
    x = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    spec[0] = 0.0
    total = spec.sum() + 1e-9
    centroid = float((freqs * spec).sum() / total)
    spread = float(np.sqrt(((freqs - centroid) ** 2 * spec).sum() / total))
    peak_idx = np.argsort(spec)[-top_k:][::-1]
    feats = []
    for idx in peak_idx:
        feats.extend([float(freqs[idx] / sample_rate), float(spec[idx] / total)])
    while len(feats) < 2 * top_k:
        feats.extend([0.0, 0.0])
    edges = np.linspace(0, freqs.max(), n_bands + 1)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        feats.append(float(spec[mask].sum() / total) if mask.any() else 0.0)
    feats.extend([centroid / sample_rate, spread / sample_rate])
    return np.asarray(feats, dtype=float)


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    R, H = len(r), len(h)
    D = np.zeros((R + 1, H + 1), dtype=int)
    D[:, 0] = np.arange(R + 1)
    D[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            D[i, j] = D[i - 1, j - 1] if r[i - 1] == h[j - 1] else 1 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[R, H]) / max(1, R)


def cer(ref, hyp):
    r, h = list(ref.replace(' ', '')), list(hyp.replace(' ', ''))
    R, H = len(r), len(h)
    D = np.zeros((R + 1, H + 1), dtype=int)
    D[:, 0] = np.arange(R + 1)
    D[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            D[i, j] = D[i - 1, j - 1] if r[i - 1] == h[j - 1] else 1 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[R, H]) / max(1, R)


def build_vocab():
    vocab = ["alice", "bob", "charlie", "one", "two", "twenty three", "ninety nine"]
    base_freqs = [240, 320, 410, 500, 620, 760, 910]
    vocab_wave_map = {i: [(base_freqs[i], 0.7), (base_freqs[i] * 2.03, 0.35)] for i in range(len(vocab))}
    return vocab, vocab_wave_map


def run(seed: int = 42) -> dict:
    vocab, vocab_wave_map = build_vocab()
    samples = []
    for i, phrase in enumerate(vocab):
        for c in ['clean', 'noisy', 'stretched']:
            for rep in range(8):
                noise, stretch = (0.01, 1.0) if c == 'clean' else ((0.12, 1.0) if c == 'noisy' else (0.05, 1.15))
                sig, sr = synth_clip(i, vocab_wave_map, noise_level=noise, stretch=stretch, seed=seed + i * 100 + rep)
                samples.append({'id': f'{i}_{c}_{rep}', 'phrase_id': i, 'transcript': phrase, 'condition': c, 'feats': extract_band_features(sig, sr)})
    df = pd.DataFrame(samples)
    cond_map = {'clean': 0.0, 'noisy': 1.0, 'stretched': 2.0}
    X = np.vstack([np.concatenate([row.feats, np.array([cond_map[row.condition]])]) for _, row in df.iterrows()])
    y = df['phrase_id'].values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.25, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    pred_transcripts = [vocab[int(p)] for p in y_pred]
    ref_transcripts = df.loc[idx_test, 'transcript'].tolist()
    conditions_test = df.loc[idx_test, 'condition'].tolist()
    acc = float(accuracy_score(y_test, y_pred))
    mean_wer = float(np.mean([wer(r, h) for r, h in zip(ref_transcripts, pred_transcripts)]))
    mean_cer = float(np.mean([cer(r, h) for r, h in zip(ref_transcripts, pred_transcripts)]))
    preview = []
    for i_idx, row_idx in enumerate(idx_test[:6]):
        preview.append({'clip_id': df.at[row_idx, 'id'], 'start_time_s': 0.0, 'end_time_s': 0.8, 'condition': conditions_test[i_idx], 'ref': ref_transcripts[i_idx], 'hyp': pred_transcripts[i_idx]})
    return {'n_samples': int(len(df)), 'n_test': int(len(idx_test)), 'n_classes': int(len(vocab)), 'word_accuracy': acc, 'wer': mean_wer, 'cer': mean_cer, 'transcript_preview': preview}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
