from sklearn.svm import OneClassSVM


class SVMDetector:
    def __init__(self, nu=0.1, kernel="rbf", gamma="scale"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)
