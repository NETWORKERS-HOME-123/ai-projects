import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.utils import check_random_state


def _sobel_filters():
    # simple 3x3 sobel kernels
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    ky = kx.T
    return kx, ky


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img, dtype=float)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def _extract_features(img: np.ndarray) -> np.ndarray:
    # features: raw pixels, sobel x, sobel y, local mean (3x3)
    kx, ky = _sobel_filters()
    sobel_x = _convolve2d(img, kx)
    sobel_y = _convolve2d(img, ky)
    # local mean via simple uniform kernel
    mean_k = np.ones((3, 3), dtype=float) / 9.0
    local_mean = _convolve2d(img, mean_k)
    # stack features as vectors per-pixel
    h, w = img.shape
    feat = np.stack([img, sobel_x, sobel_y, local_mean], axis=-1)
    return feat.reshape((-1, feat.shape[-1]))  # (num_pixels, channels)


def _affine_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Match first and second order statistics via affine transform y = A x + b
    # Closed-form: choose A as (cov_t)^0.5 (cov_s)^-0.5 but for stability use diag of stds
    # For small educational code we match per-feature mean and variance (diagonal transform).
    eps = 1e-8
    mean_s = source.mean(axis=0)
    mean_t = target.mean(axis=0)
    std_s = source.std(axis=0) + eps
    std_t = target.std(axis=0) + eps
    A = (std_t / std_s)
    b = mean_t - A * mean_s
    transformed = source * A + b
    return transformed


def run(seed: int = 42) -> Dict[str, Any]:
    """Run a small, deterministic style-transfer-like demo and return metrics.

    This is a toy, closed-form statistic-matching transfer on 8x8 greyscale digits.
    """
    rs = check_random_state(seed)

    data = load_digits()
    images = data.images.astype(float)  # range roughly 0..16

    # pick two distinct images reproducibly: content and style
    idx = rs.choice(len(images), size=2, replace=False)
    content = images[idx[0]].copy()
    style = images[idx[1]].copy()

    # normalise to 0..1 for computations
    content_norm = (content - content.min()) / (content.max() - content.min() + 1e-12)
    style_norm = (style - style.min()) / (style.max() - style.min() + 1e-12)

    # feature extraction per-pixel
    feats_content = _extract_features(content_norm)
    feats_style = _extract_features(style_norm)

    # compute affine per-feature transform from content->style statistics
    feats_trans = _affine_match(feats_content, feats_style)

    # reconstruct an image from transformed features by taking the first channel (approx intensity)
    h, w = content.shape
    recon = feats_trans[:, 0].reshape((h, w))

    # blend with original to preserve content structure; alpha chosen to preserve content
    alpha = 0.85
    output = alpha * content_norm + (1.0 - alpha) * recon

    # clamp
    output = np.clip(output, 0.0, 1.0)

    # metrics: means for provenance, MSE between content and output, Pearson corr of flattened images
    content_mean = float(content_norm.mean())
    style_mean = float(style_norm.mean())

    mse = float(np.mean((content_norm - output) ** 2))
    # Pearson correlation
    c = content_norm.flatten()
    o = output.flatten()
    # safe Pearson
    cc = float(np.corrcoef(c, o)[0, 1])

    # additional small metric: relative change energy
    rel_change = float(np.linalg.norm(o - c) / (np.linalg.norm(c) + 1e-12))

    # Return JSON-serialisable metrics and tiny provenance info.
    return {
        "content_index": int(idx[0]),
        "style_index": int(idx[1]),
        "content_image_mean": content_mean,
        "style_image_mean": style_mean,
        "content_mse": mse,
        "content_corr": cc,
        "relative_change": rel_change,
        "alpha": float(alpha),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
