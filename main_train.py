import argparse, os
from src.utils import ensure_dir, save_json, save_model
from src.preprocess import load_data, merge_identity, basic_clean, basic_impute, target_encode_train, create_holdout
from src.feature_engineering import log_transform_amount, create_time_features, card_agg_features, reduce_V_features_pca, select_features
from src.train_lightgbm import train_lgbm_kfold, evaluate_binary
from src.anomaly_isoforest import train_isolation_forest
import pandas as pd
import joblib
import json

def main(data_dir, out_dir, models_dir):
    ensure_dir(out_dir); ensure_dir(models_dir)
    print('Loading data...')
    tx, idf = load_data(data_dir)
    print('Merging identity...')
    df = merge_identity(tx, idf)
    print('Basic clean...')
    df = basic_clean(df)
    print('Impute...')
    df = basic_impute(df)
    print('Feature engineering...')
    df = log_transform_amount(df)
    df = create_time_features(df)
    df = card_agg_features(df)
    df = reduce_V_features_pca(df, n_components=20)
    # create label if missing (use existing isFraud)
    if 'isFraud' not in df.columns:
        raise ValueError('isFraud column not found in dataset. Provide labeled data.')
    print('Target encoding a few top categorical cols if present...')
    cat_cols = [c for c in df.columns if c.startswith('card') or c.startswith('addr') or c.startswith('ProductCD')]
    cat_cols = [c for c in cat_cols if df[c].dtype == 'object' or df[c].nunique() < 200]
    if len(cat_cols) > 0:
        df, enc = target_encode_train(df, cat_cols, target_col='isFraud')
        joblib.dump(enc, os.path.join(models_dir, 'target_encoder.joblib'))
    print('Creating train/val split...')
    train, val = create_holdout(df, test_size=0.2)
    y_train = train['isFraud']
    y_val = val['isFraud']
    # select features
    features = select_features(train, max_features=120)
    print('Selected features count:', len(features))

    joblib.dump(features, f"{models_dir}/selected_features.pkl")
    X_train = train[features]
    X_val = val[features]
    print('Training LightGBM with K-Fold...')
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }
    models, oof, overall_auc = train_lgbm_kfold(pd.concat([X_train, X_val]).reset_index(drop=True),
                                               pd.concat([y_train, y_val]).reset_index(drop=True),
                                               features, n_splits=5, params=params)
    # save models (ensemble)
    for i, m in enumerate(models):
        save_model(m, os.path.join(models_dir, f'lightgbm_fold{i+1}.pkl'))
    # evaluate on holdout val (use oof or predict)
    print('Training IsolationForest for anomaly detection (using train only)...')
    iso, iso_scores = train_isolation_forest(X_train.fillna(-1), contamination=0.01)
    save_model(iso, os.path.join(models_dir, 'isoforest.joblib'))
    # Final metrics (simple)
    from sklearn.metrics import roc_auc_score
    # compute predictions on val using average of models
    import numpy as np
    preds = 0
    for m in models:
        preds += m.predict(val[features], num_iteration=m.best_iteration) / len(models)
    auc_val = roc_auc_score(y_val, preds)
    metrics = {
        'oof_auc': float(overall_auc),
        'val_auc': float(auc_val)
    }
    save_json(metrics, os.path.join(out_dir, 'evaluation_metrics.json'))
    print('Done. Metrics saved to outputs. Models saved to models/.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--out_dir', default='outputs')
    parser.add_argument('--models_dir', default='models')
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.models_dir)
