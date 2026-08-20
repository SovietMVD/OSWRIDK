import numpy as np
from sklearn.neighbors import NearestNeighbors


class INNEDetector:
    def __init__(self, n_estimators=100, sample_size=256, random_state=42):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.rng = np.random.RandomState(random_state)

    def fit(self, X):
        self.X = X
        self.n = X.shape[0]
        self.sample_size = min(self.sample_size, self.n)
        return self

    def score(self, X):
        scores = np.zeros(X.shape[0])
        for _ in range(self.n_estimators):
            idx = self.rng.choice(self.n, self.sample_size, replace=False)
            sub = self.X[idx]
            nbrs = NearestNeighbors(n_neighbors=2).fit(sub)
            dist, _ = nbrs.kneighbors(X)
            scores += dist[:, 1]
        return scores / self.n_estimators
