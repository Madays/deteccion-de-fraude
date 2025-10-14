import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np

def train_lgbm_kfold(X, y, features, n_splits=5, params=None, seed=42):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = []
    oof = np.zeros(len(y))
    fold = 0
    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        X_train, X_val = X.iloc[train_idx][features], X.iloc[val_idx][features]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
        clf = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[train_set, val_set],
            callbacks=callbacks
        )

        pred = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof[val_idx] = pred
        models.append(clf)
        print(f"Fold {fold} AUC:", roc_auc_score(y_val, pred))
    overall_auc = roc_auc_score(y, oof)
    print("OOF AUC:", overall_auc)
    return models, oof, overall_auc

def train_lgbm_kfold_stratified(X, y, features, n_splits=5, params=None, seed=42):
    """
    Entrena modelos LightGBM usando StratifiedKFold para mantener la proporción de clases
    en cada fold (importante para datasets desbalanceados como fraude).
    Retorna:
      - Lista de modelos entrenados (uno por fold)
      - Predicciones OOF (out-of-fold)
      - AUC promedio global
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = []
    oof = np.zeros(len(y))
    fold = 0

    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        print(f"=== Fold {fold}/{n_splits} ===")
        X_train, X_val = X.iloc[train_idx][features], X.iloc[val_idx][features]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Crear datasets de LightGBM
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val)

        # Entrenamiento con early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]

        clf = lgb.train(
            params,
            train_set,
            num_boost_round=5000,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        # Guardar predicciones de validación (OOF)
        pred = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof[val_idx] = pred
        models.append(clf)

        fold_auc = roc_auc_score(y_val, pred)
        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    # AUC global usando todas las predicciones OOF
    overall_auc = roc_auc_score(y, oof)
    print(f"✅ OOF AUC (Stratified {n_splits}-Fold): {overall_auc:.6f}")
    return models, oof, overall_auc

def evaluate_binary(y_true, y_pred_prob, threshold=0.5):
    preds = (y_pred_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    return {'precision': prec, 'recall': rec, 'f1': f1}
