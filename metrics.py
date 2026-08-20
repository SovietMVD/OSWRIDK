import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score


def auroc(labels, scores):
    return float(roc_auc_score(labels, scores))


def aupr(labels, scores):
    return float(average_precision_score(labels, scores))


def fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def closed_set_f1(labels, scores, class_scores, tau, average="macro"):
    preds = []
    for s, cs in zip(scores, class_scores):
        if s < tau:
            preds.append(-1)
        else:
            preds.append(int(np.argmax(cs)))
    return float(f1_score(labels, preds, average=average, zero_division=0))
