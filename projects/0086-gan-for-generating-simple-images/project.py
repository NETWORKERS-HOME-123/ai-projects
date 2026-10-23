import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(y):
    return y * (1 - y)

class SimpleMLP:
    def __init__(self, sizes, seed):
        rng = np.random.RandomState(seed)
        self.sizes = sizes
        self.W = [rng.normal(0, 0.1, (n_out, n_in)) for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((n_out,)) for n_out in sizes[1:]]
    def forward(self, x):
        activations = [x]
        for i in range(len(self.W)):
            x = np.dot(x, self.W[i].T) + self.b[i]
            x = np.tanh(x) if i < len(self.W) - 1 else sigmoid(x)
            activations.append(x)
        return activations
    def zero_grad_like(self):
        return ([np.zeros_like(w) for w in self.W], [np.zeros_like(b) for b in self.b])
    def apply_grads(self, dW, db, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

def binary_cross_entropy(pred, target):
    eps = 1e-8
    p = np.clip(pred, eps, 1 - eps)
    return - (target * np.log(p) + (1 - target) * np.log(1 - p))

def run(seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    data = load_digits()
    X = data.images.reshape(len(data.images), -1) / 16.0
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed, stratify=data.target)
    latent_dim = 10
    img_dim = X_train.shape[1]
    G = SimpleMLP([latent_dim, 64, img_dim], seed=seed + 1)
    D = SimpleMLP([img_dim, 64, 1], seed=seed + 2)
    n_epochs = 800
    batch_size = 64
    lr_d = 0.01
    lr_g = 0.01
    def sample_z(n):
        return rng.normal(0, 1, (n, latent_dim))
    for epoch in range(n_epochs):
        idx = rng.choice(len(X_train), size=min(batch_size, len(X_train)), replace=False)
        real = X_train[idx]
        n_real = real.shape[0]
        z = sample_z(n_real)
        aG = G.forward(z)
        fake = aG[-1]
        # Discriminator update
        Xd = np.vstack([real, fake])
        yd = np.concatenate([np.ones(n_real), np.zeros(n_real)])[:, None]
        aD = D.forward(Xd)
        pred = aD[-1]
        delta = (pred - yd)
        dW, db = D.zero_grad_like()
        for i in range(len(D.W) - 1, -1, -1):
            inp = aD[i]
            dW[i] += (delta.T @ inp) / inp.shape[0]
            db[i] += delta.mean(axis=0)
            if i > 0:
                delta = (delta @ D.W[i]) * (1 - aD[i] ** 2)
        D.apply_grads(dW, db, lr_d)
        # Generator update: push D(G(z)) toward 1
        aD_fake = D.forward(fake)
        pred_fake = aD_fake[-1]
        yg = np.ones_like(pred_fake)
        delta = (pred_fake - yg)
        # backprop through D to get grad wrt fake inputs
        for i in range(len(D.W) - 1, -1, -1):
            w = D.W[i]
            if i == 0:
                grad_input = delta @ w
            else:
                delta = (delta @ w) * (1 - aD_fake[i] ** 2)
        grad_fake = grad_input
        aG = G.forward(z)
        dWg, dbg = G.zero_grad_like()
        delta_g = grad_fake
        for i in range(len(G.W) - 1, -1, -1):
            inp = aG[i]
            if i == len(G.W) - 1:
                grad_act = delta_g * sigmoid_grad(aG[-1])
            else:
                grad_act = delta_g * (1 - aG[i+1] ** 2)
            dWg[i] += (grad_act.T @ inp) / inp.shape[0]
            dbg[i] += grad_act.mean(axis=0)
            if i > 0:
                delta_g = grad_act @ G.W[i]
        G.apply_grads(dWg, dbg, lr_g)
        if (epoch + 1) % 200 == 0:
            lr_d *= 0.8
            lr_g *= 0.8
    # Evaluation
    n_eval = 200
    z_eval = rng.normal(0, 1, (n_eval, latent_dim))
    fake_eval = G.forward(z_eval)[-1]
    Xd_test = np.vstack([X_test[:n_eval], fake_eval[:n_eval]])
    yd_test = np.concatenate([np.ones(min(len(X_test), n_eval)), np.zeros(n_eval)])[:, None]
    pred_test = D.forward(Xd_test)[-1]
    pred_label = (pred_test >= 0.5).astype(int)
    disc_acc = (pred_label.flatten() == yd_test.flatten()).mean()
    def pairwise_mean_dist(A):
        diff = A[:, None, :] - A[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        return d[np.triu_indices_from(d, k=1)].mean()
    gen_diversity = float(pairwise_mean_dist(fake_eval))
    from numpy.linalg import norm
    dists = [np.min(norm(X_train - v, axis=1)) for v in fake_eval]
    mean_nn = float(np.mean(dists))
    return {"discriminator_accuracy": float(disc_acc), "generator_diversity": float(gen_diversity), "mean_nearest_train_dist": float(mean_nn)}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
