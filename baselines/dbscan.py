import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class DBSCANDetector:
    def __init__(self, eps=45, min_samples=1):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.X = X
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(X)
        self.labels = labels
        values, counts = np.unique(labels, return_counts=True)
        self.core_label = values[np.argmax(counts)]
        return self

    def score(self, X):
        core = self.X[self.labels == self.core_label]
        if len(core) == 0:
            core = self.X
        nbrs = NearestNeighbors(n_neighbors=1).fit(core)
        dist, _ = nbrs.kneighbors(X)
        return dist.ravel()
