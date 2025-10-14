import joblib
from sklearn.ensemble import IsolationForest

def train_isolation_forest(X, contamination=0.01, random_state=42):
    """
    X: DataFrame con features ya procesadas (numéricas) - debe corresponder a selected_features
    """
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X)
    scores = model.decision_function(X)
    return model, scores
