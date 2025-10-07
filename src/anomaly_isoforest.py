from sklearn.ensemble import IsolationForest
import numpy as np

def train_isolation_forest(X, contamination=0.01, random_state=42):
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    iso.fit(X)
    scores = -iso.decision_function(X)  # higher = more anomalous
    return iso, scores

def score_with_isoforest(iso, X):
    return -iso.decision_function(X)
