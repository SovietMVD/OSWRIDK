import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from baselines import DETECTORS


def run_benchmark(class_features, detector_name, num_categories=100, seed=42):
    rng = np.random.RandomState(seed)
    detector_cls = DETECTORS[detector_name]
    cats = list(range(len(class_features)))
    if len(cats) > num_categories:
        cats = rng.choice(cats, num_categories, replace=False)

    y_true_all = []
    y_pred_all = []
    for c in cats:
        normal = class_features[c]
        others = np.vstack([class_features[j] for j in range(len(class_features)) if j != c])
        anomaly_idx = rng.choice(len(others), 3, replace=False)
        anomalies = others[anomaly_idx]
        X = np.vstack([normal, anomalies])
        y_true = np.array([0] * len(normal) + [1] * 3)

        detector = detector_cls()
        detector.fit(normal)
        scores = detector.score(X)
        threshold = np.percentile(scores, 100 * (1 - 3.0 / len(normal)))
        y_pred = (scores > threshold).astype(int)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return {
        "Precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
        "Recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
        "F1": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
    }
