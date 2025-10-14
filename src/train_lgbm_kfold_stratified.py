import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def train_lgbm_kfold(X, y, features, params, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = []
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr][features], X.iloc[va][features]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dtrain,dvalid],
                          early_stopping_rounds=50, verbose_eval=100)
        preds = model.predict(X_va, num_iteration=model.best_iteration)
        oof[va] = preds
        models.append(model)
    overall_auc = roc_auc_score(y, oof)
    return models, oof, overall_auc
