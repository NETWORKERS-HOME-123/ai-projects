import json, math
from typing import Dict, List
import numpy as np
from sklearn.model_selection import train_test_split

def build_vocab(docs: List[str]):
    chars = sorted(set(''.join(docs)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos

def one_hot(i, n):
    v = np.zeros(n, dtype=np.float32); v[i] = 1.0; return v

class TinyRNN:
    def __init__(self, vocab_size, hidden_size, seed):
        rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size; self.hidden_size = hidden_size
        self.Wxh = rng.randn(hidden_size, vocab_size) * 0.1
        self.Whh = rng.randn(hidden_size, hidden_size) * 0.1
        self.bh = np.zeros((hidden_size, ))
        self.Why = rng.randn(vocab_size, hidden_size) * 0.1
        self.by = np.zeros((vocab_size, ))
    def step(self, x_onehot, h_prev):
        h = np.tanh(self.Wxh @ x_onehot + self.Whh @ h_prev + self.bh)
        y = self.Why @ h + self.by
        y = y - np.max(y); expy = np.exp(y); probs = expy / np.sum(expy)
        return probs, h

def seq_to_pairs(s: str, stoi):
    ids = [stoi[c] for c in s]
    return [(ids[i], ids[i+1]) for i in range(len(ids)-1)]

def cross_entropy(probs, target_id):
    return -math.log(max(probs[target_id], 1e-12))

def sample_from_probs(probs, rng, temperature=1.0):
    if temperature <= 0:
        return int(np.argmax(probs))
    logits = np.log(np.maximum(probs, 1e-12)) / temperature
    logits = logits - np.max(logits); exp = np.exp(logits); p = exp / np.sum(exp)
    return int(rng.choice(len(p), p=p))

def generate_text(model: TinyRNN, seed_text: str, stoi, itos, length: int, temperature: float, seed: int):
    rng = np.random.RandomState(seed)
    h = np.zeros(model.hidden_size)
    for ch in seed_text:
        if ch in stoi:
            x = one_hot(stoi[ch], model.vocab_size); probs, h = model.step(x, h)
    out = list(seed_text)
    curr_id = stoi[out[-1]] if out and out[-1] in stoi else rng.randint(0, model.vocab_size)
    for _ in range(length):
        x = one_hot(curr_id, model.vocab_size); probs, h = model.step(x, h)
        curr_id = sample_from_probs(probs, rng, temperature); out.append(itos[curr_id])
    return ''.join(out)

def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)
    docs = [
        "the river flows by the old mill",
        "a child picks a bright red apple",
        "the night sky had a single bright star",
        "an old book lay open on the wooden desk",
        "the cat slept on the warm windowsill",
        "small boats drifted near the rocky shore",
        "a bell rang softly at morning",
        "the kettle sang when the water boiled",
        "the gardener planted seeds in neat rows",
        "a paper plane flew across the classroom"
    ]
    train_docs, val_docs = train_test_split(docs, test_size=0.3, random_state=seed)
    stoi, itos = build_vocab(docs); V = len(stoi)
    train_pairs = []
    for d in train_docs: train_pairs.extend(seq_to_pairs(d, stoi))
    val_pairs = []
    for d in val_docs: val_pairs.extend(seq_to_pairs(d, stoi))
    hidden_size = 32; lr = 0.5; epochs = 40
    model = TinyRNN(V, hidden_size, seed)
    for epoch in range(epochs):
        rng.shuffle(train_pairs)
        for x_id, y_id in train_pairs:
            x = one_hot(x_id, V); h_prev = np.zeros(hidden_size)
            probs, h = model.step(x, h_prev)
            dy = probs.copy(); dy[y_id] -= 1.0
            dWhy = np.outer(dy, h); dby = dy
            dh = model.Why.T @ dy
            dt = dh * (1 - h * h)
            dWxh = np.outer(dt, x); dWhh = np.outer(dt, h_prev); dbh = dt
            model.Why -= lr * dWhy; model.by -= lr * dby
            model.Wxh -= lr * dWxh; model.Whh -= lr * dWhh; model.bh -= lr * dbh
        lr *= 0.99
    total_loss = 0.0; correct = 0
    for x_id, y_id in val_pairs:
        x = one_hot(x_id, V); h_prev = np.zeros(hidden_size)
        probs, h = model.step(x, h_prev)
        total_loss += cross_entropy(probs, y_id)
        if int(np.argmax(probs)) == y_id: correct += 1
    val_loss = total_loss / max(1, len(val_pairs)); val_acc = correct / max(1, len(val_pairs))
    seed_text = "the"
    sample_t05 = generate_text(model, seed_text, stoi, itos, length=60, temperature=0.5, seed=seed)
    sample_t1 = generate_text(model, seed_text, stoi, itos, length=60, temperature=1.0, seed=seed + 1)
    sample_t2 = generate_text(model, seed_text, stoi, itos, length=60, temperature=2.0, seed=seed + 2)
    return {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "sample_temperature_0.5": sample_t05,
        "sample_temperature_1.0": sample_t1,
        "sample_temperature_2.0": sample_t2,
        "vocab_size": int(V),
        "epochs": int(epochs)
    }

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
