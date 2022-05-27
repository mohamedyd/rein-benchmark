import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support


def compute_distances(X_train, X_test):
    dists = np.array([np.sqrt(np.sum((X_train - x_test)**2, axis=1)) for x_test in X_test])
    return dists

def majority_vote(A):
    counter = Counter(A)
    major = counter.most_common(1)[0][0]
    return int(major)

class KNN(object):
    """docstring for KNNEvaluator"""
    def __init__(self, n_neighbors=3):
        super(KNN).__init__()
        self.K = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        dists = compute_distances(self.X_train, X_test)
        self.sim = 1 / (1 + dists)
        order = np.argsort(-self.sim, kind="stable", axis=1)
        top_K_idx = order[:, :self.K]
        top_K = self.y_train[top_K_idx]
        pred = np.array([majority_vote(top) for top in top_K])
        return pred

    def score(self, X_test, y_test):
        pred = self.predict(X_test)
        acc = np.mean(pred == y_test)
        return acc
    
    # new function to get precision, recall and f1
    def prf(self, X_test, y_test):
        y_pred_test = self.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
        return p, r, f1