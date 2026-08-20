import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.spatial import KDTree


def ik_feature(points, centers, t, center_indices):
    n = points.shape[0]
    blocks = []
    for idx in center_indices:
        tree = KDTree(centers[idx])
        _, nn = tree.query(points, k=1)
        block = csr_matrix((np.ones(n), (np.arange(n), nn.reshape(-1))), shape=(n, len(idx)))
        blocks.append(block)
    return hstack(blocks)


def class_threshold(phi, t):
    m = phi.shape[0]
    if m < 2:
        return 1.0
    s = (phi @ phi.T).toarray() / t
    off = s.sum() - m
    return float(off / (m * (m - 1)))


def point_class_similarity(fx, phi, t):
    mean_phi = phi.mean(axis=0)
    val = fx @ mean_phi.T
    return float(np.asarray(val).reshape(-1)[0]) / t


class IDK:
    def __init__(self, psi_ratio=1.0 / 3.0, t=200, seed=42):
        self.psi_ratio = psi_ratio
        self.t = t
        self.rng = np.random.RandomState(seed)

    def fit(self, class_features):
        self.classes = []
        for feats in class_features:
            m = feats.shape[0]
            psi = max(1, int(m * self.psi_ratio))
            idx = [self.rng.choice(m, psi, replace=False) for _ in range(self.t)]
            phi = ik_feature(feats, feats, self.t, idx)
            self.classes.append({"feats": feats, "phi": phi, "idx": idx})
        self.mu = [class_threshold(c["phi"], self.t) for c in self.classes]
        self.tau = min(self.mu)

    def score(self, x):
        scores = []
        for c in self.classes:
            fx = ik_feature(x.reshape(1, -1), c["feats"], self.t, c["idx"])
            scores.append(point_class_similarity(fx, c["phi"], self.t))
        return scores

    def predict(self, x):
        scores = self.score(x)
        s = max(scores)
        if s < self.tau:
            return -1, s
        return int(np.argmax(scores)), s
