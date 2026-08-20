import os

import numpy as np
import torch

import config
from dataset import build_transform, collect_ood, collect_samples
from preprocess import preprocess_path
from dbn import DBN
from idk import IDK
from metrics import auroc, aupr, fpr95, closed_set_f1


def _device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract(model, paths, device):
    model.eval()
    transform = build_transform(config.INPUT_SIZE)
    feats = []
    for path in paths:
        full, patch = preprocess_path(path, config.PATCH_SIZE)
        gf = transform(full).unsqueeze(0).to(device)
        lf = transform(patch).unsqueeze(0).to(device)
        out = model(gf, lf)
        feat = out[1] if isinstance(out, tuple) else out
        feats.append(feat.cpu().numpy().reshape(-1))
    return np.array(feats)


def evaluate(model_path, data_root):
    device = _device()
    train_root = os.path.join(data_root, config.TRAIN_DIR)
    test_root = os.path.join(data_root, config.TEST_DIR)
    ood_root = os.path.join(data_root, config.OOD_DIR)

    train_samples = collect_samples(train_root)
    test_samples = collect_samples(test_root)
    ood_paths = collect_ood(ood_root)

    num_classes = max(l for _, l in train_samples) + 1
    model = DBN(config.BACKBONE, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)

    train_paths = [p for p, _ in train_samples]
    test_paths = [p for p, _ in test_samples]

    train_feats = extract(model, train_paths, device)
    test_feats = extract(model, test_paths, device)
    ood_feats = extract(model, ood_paths, device)

    train_labels = np.array([l for _, l in train_samples])
    class_features = [train_feats[train_labels == c] for c in range(num_classes)]

    detector = IDK(psi_ratio=config.PSI_RATIO, t=config.T, seed=config.SEED)
    detector.fit(class_features)

    all_feats = np.vstack([test_feats, ood_feats])
    labels = np.array([1] * len(test_feats) + [0] * len(ood_feats))
    scores = []
    class_scores = []
    for feat in all_feats:
        cs = detector.score(feat)
        class_scores.append(cs)
        scores.append(max(cs))
    scores = np.array(scores)
    class_scores = np.array(class_scores)

    id_scores = scores[:len(test_feats)]
    id_class_scores = class_scores[:len(test_feats)]
    id_true = np.array([l for _, l in test_samples])

    return {
        "AUROC": auroc(labels, scores),
        "AUPR": aupr(labels, scores),
        "FPR95": fpr95(labels, scores),
        "F1": closed_set_f1(id_true, id_scores, id_class_scores, detector.tau),
    }
