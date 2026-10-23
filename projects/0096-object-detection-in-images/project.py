import json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


def bbox_of_image(img: np.ndarray, threshold: float = 8.0) -> Tuple[int,int,int,int]:
    """Compute minimal bounding box of pixels > threshold in an 8x8 image.
    Returns (x_min, y_min, x_max, y_max) in pixel coords (0..7). If no pixel above
    threshold, return full image box.
    """
    h, w = img.shape
    mask = img > threshold
    if not mask.any():
        return 0, 0, w - 1, h - 1
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = areaA + areaB - inter
    if union == 0:
        return 0.0
    return inter / union


def average_precision_at_iou(gts: List[Tuple[int,Tuple[int,int,int,int]]],
                             preds: List[Tuple[float,int,Tuple[int,int,int,int]]],
                             iou_threshold: float) -> float:
    """Compute AP for single-class per-image (one gt per image) style.
    gts: list of (image_id, box)
    preds: list of (score, image_id, box)
    This is simplified: at most one ground-truth and one prediction per image.
    """
    if len(gts) == 0:
        return 0.0
    gt_map = {img: box for img, box in gts}
    preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
    tp = []
    fp = []
    matched = set()
    for score, img_id, box in preds_sorted:
        if img_id in gt_map and img_id not in matched and iou(box, gt_map[img_id]) >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched.add(img_id)
        else:
            tp.append(0)
            fp.append(1)
    tp_cum = np.cumsum(tp).astype(float)
    fp_cum = np.cumsum(fp).astype(float)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)
    recalls = tp_cum / (len(gts) + 1e-12)
    # AP: integrate precision over recall by taking max-precision at each recall step
    if len(precisions) == 0:
        return 0.0
    # Use 11-point interpolation (classic, simple)
    recall_levels = np.linspace(0, 1, 11)
    ap = 0.0
    for r in recall_levels:
        p = precisions[recalls >= r]
        if p.size == 0:
            p_val = 0.0
        else:
            p_val = p.max()
        ap += p_val / 11.0
    return float(ap)


def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)
    digits = load_digits()
    X_imgs = digits.images.copy()  # (n_samples, 8, 8)
    y = digits.target.copy()
    n = X_imgs.shape[0]
    # Build one bbox per image from bright pixels
    boxes = np.array([bbox_of_image(img) for img in X_imgs])  # shape (n,4)
    # Create a simple feature vector: flattened pixels and normalized bbox coords
    flat = X_imgs.reshape(n, -1).astype(float)
    bbox_norm = boxes / 7.0  # scale to 0..1
    features = np.hstack([flat, bbox_norm])

    # Create a stratified split so class distribution is preserved
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_idx, test_idx = next(sss.split(features, y))

    X_train, X_test = features[train_idx], features[test_idx]
    y_train_cls, y_test_cls = y[train_idx], y[test_idx]
    boxes_train, boxes_test = boxes[train_idx], boxes[test_idx]
    imgs_test = X_imgs[test_idx]

    # Train a classifier for class and a regressor for bbox coords (predict 4 numbers)
    cls = RandomForestClassifier(n_estimators=100, random_state=seed)
    reg = RandomForestRegressor(n_estimators=100, random_state=seed)
    cls.fit(X_train, y_train_cls)
    reg.fit(X_train, boxes_train)

    # Predict on test set
    y_pred_cls = cls.predict(X_test)
    y_pred_proba = cls.predict_proba(X_test)
    # confidence = probability of predicted class
    confidences = np.max(y_pred_proba, axis=1)
    boxes_pred = np.round(reg.predict(X_test)).astype(int)
    # Clip boxes to image coords
    boxes_pred = np.clip(boxes_pred, 0, 7)
    # Build lists for AP calculation: image ids are indices in test set
    gts = [(int(i), tuple(map(int, boxes_test[idx]))) for idx, i in enumerate(range(len(test_idx)))]
    preds = [(float(confidences[i]), int(i), tuple(map(int, boxes_pred[i]))) for i in range(len(test_idx))]

    map50 = average_precision_at_iou(gts, preds, 0.5)
    # Also compute mean IoU between predicted and gt boxes
    ious = [iou(boxes_test[i], boxes_pred[i]) for i in range(len(boxes_test))]
    mean_iou = float(np.mean(ious))
    acc = float(accuracy_score(y_test_cls, y_pred_cls))

    # Also report AP@0.75 which will typically be lower
    map75 = average_precision_at_iou(gts, preds, 0.75)

    # Provide a simple editable viewer output: a small DataFrame (as dict) of first 10
    viewer_rows = []
    for i in range(min(10, len(test_idx))):
        viewer_rows.append({
            "image_id": int(i),
            "pred_box": tuple(int(x) for x in boxes_pred[i]),
            "gt_box": tuple(int(x) for x in boxes_test[i]),
            "pred_class": int(y_pred_cls[i]),
            "gt_class": int(y_test_cls[i]),
            "confidence": float(confidences[i])
        })

    return {
        "map50": map50,
        "map75": map75,
        "mean_iou": mean_iou,
        "class_accuracy": acc,
        "viewer_sample_count": len(viewer_rows),
        "viewer_sample": viewer_rows
    }


if __name__ == "__main__":
    print(json.dumps(run(42), indent=2))
