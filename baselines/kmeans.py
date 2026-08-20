from sklearn.cluster import KMeans


class KMeansDetector:
    def __init__(self, n_clusters=1, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X):
        self.model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X)
        return self

    def score(self, X):
        return self.model.transform(X).min(axis=1)
