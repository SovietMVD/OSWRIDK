from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, n_estimators=100, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)
