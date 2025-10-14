import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from src import preprocess, feature_engineering, utils
from src.anomaly_isoforest import train_isolation_forest
import lightgbm as lgb

# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -------------------------
# 1) Cargar datos y unir identity (si existe)
# -------------------------
print("[1/6] Cargando datos...")
tx_path = os.path.join(DATA_DIR, "train_transaction.csv")
if not os.path.exists(tx_path):
    raise FileNotFoundError(f"No se encontró {tx_path}")
df = pd.read_csv(tx_path)
idf_path = os.path.join(DATA_DIR, "train_identity.csv")
if os.path.exists(idf_path):
    print("   → Merging identity...")
    idf = pd.read_csv(idf_path)
    df = df.merge(idf, on="TransactionID", how="left")

# -------------------------
# 2) Limpieza e imputación básicas
# -------------------------
print("[2/6] Preprocesando (clean + impute)...")
df = preprocess.basic_clean(df)
df = preprocess.basic_impute(df)

# Eliminar filas sin etiqueta de target (si existieran)
if 'isFraud' in df.columns:
    missing_targets = df['isFraud'].isna().sum()
    if missing_targets > 0:
        print(f"   ⚠️ {missing_targets} filas sin etiqueta en dataset - serán eliminadas.")
        df = df[df['isFraud'].notna()].reset_index(drop=True)
else:
    raise ValueError("isFraud no está en las columnas. Necesitas datos etiquetados para entrenar.")

# -------------------------
# 3) Crear features temporales (para split temporal)
# -------------------------
print("[3/6] Creando features temporales...")
df = feature_engineering.create_time_features(df)  # añade 'day' y 'hour' si existe TransactionDT

# -------------------------
# 4) Time-based split: train_base / holdout_val
# -------------------------
cutoff_day = df['day'].quantile(0.8)
train_base_raw = df[df['day'] < cutoff_day].reset_index(drop=True)
holdout_raw = df[df['day'] >= cutoff_day].reset_index(drop=True)
print(f"   → Time split: train {len(train_base_raw)} rows, holdout {len(holdout_raw)} rows (cutoff day {cutoff_day:.3f})")

# -------------------------
# 5) Determinar columnas categoricas para target encoding (usamos train_base)
# -------------------------
# Limitamos a categorías razonables para evitar codificar IDs
cat_cols = [c for c in train_base_raw.columns if train_base_raw[c].dtype == 'object' and train_base_raw[c].nunique() < 200]
print(f"   → Columnas categóricas a target-encode (train_base): {len(cat_cols)}")

# -------------------------
# 6) Fit encoder_full y PCA_full sobre train_base (solo para holdout y predict)
# -------------------------
print("[4/6] Ajustando transformadores globales sobre train_base (encoder_full + pca_full)...")
encoder_full, train_base_enc = preprocess.fit_target_encoder(train_base_raw.copy(), cat_cols=cat_cols, target_col='isFraud')
joblib.dump(encoder_full, os.path.join(MODELS_DIR, "target_encoder_full.joblib"))
print("   → encoder_full guardado: models/target_encoder_full.joblib")

train_base_enc = feature_engineering.reduce_V_features_pca(train_base_enc, n_components=20, training=True, model_path=os.path.join(MODELS_DIR, "pca_full.joblib"))
print("   → pca_full guardado: models/pca_full.joblib")

# Aplicar mismos transformadores al holdout (para evaluación final)
holdout_enc = preprocess.apply_target_encoding(holdout_raw.copy(), encoder_full, cat_cols)
holdout_enc = feature_engineering.reduce_V_features_pca(holdout_enc, n_components=20, training=False, model_path=os.path.join(MODELS_DIR, "pca_full.joblib"))

# -------------------------
# 7) Selección de features basada en train_base_enc (no usar holdout)
# -------------------------
print("[5/6] Seleccionando features (basado en train_base_enc)...")
features = feature_engineering.select_features(train_base_enc, max_features=300)
joblib.dump(features, os.path.join(MODELS_DIR, "selected_features.pkl"))
print(f"   → {len(features)} features seleccionadas y guardadas.")

# -------------------------
# 8) Stratified K-Fold CV dentro de train_base -> obtener OOF sin fuga
#    (para OOF: volvemos a rehacer encoding+PCA por fold para evitar leak)
# -------------------------
print("[6/6] CV estratificado dentro de train_base (fold-wise encoding + PCA para evitar leakage)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n = len(train_base_raw)
oof_preds = np.zeros(n)
models = []

X_indices = np.arange(n)
y_base = train_base_raw['isFraud'].values

lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 128,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'min_data_in_leaf': 30,
    'verbosity': -1,
    'seed': 42,
    'n_jobs': -1
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_indices, y_base), 1):
    print(f"\n--- Fold {fold} ---")
    tr_raw = train_base_raw.iloc[tr_idx].reset_index(drop=True)
    va_raw = train_base_raw.iloc[va_idx].reset_index(drop=True)

    # 1) Fit encoder on train_fold only, apply to train_fold and val_fold
    enc_fold, tr_enc = preprocess.fit_target_encoder(tr_raw.copy(), cat_cols=cat_cols, target_col='isFraud')
    va_enc = preprocess.apply_target_encoding(va_raw.copy(), enc_fold, cat_cols)

    # 2) Fit PCA on train_fold only, apply to train/val (use same n_components as pca_full)
    tr_enc = feature_engineering.reduce_V_features_pca(tr_enc, n_components=20, training=True, model_path=os.path.join(MODELS_DIR, f"pca_fold{fold}.joblib"))
    va_enc = feature_engineering.reduce_V_features_pca(va_enc, n_components=20, training=False, model_path=os.path.join(MODELS_DIR, f"pca_fold{fold}.joblib"))

    # 3) Keep only 'features' (si faltan, rellenar con 0)
    for df_ in (tr_enc, va_enc):
        missing = [c for c in features if c not in df_.columns]
        if missing:
            for c in missing:
                df_[c] = 0.0
    X_tr = tr_enc[features].fillna(0)
    X_va = va_enc[features].fillna(0)
    y_tr = tr_enc['isFraud'].values
    y_va = va_enc['isFraud'].values

    # 4) Train LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    model = lgb.train(
    params=lgb_params,
    train_set=dtrain,
    num_boost_round=4000,
    valid_sets=[dtrain, dvalid],
    valid_names=['train','valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200)
    ]
    )

    models.append(model)
    joblib.dump(model, os.path.join(MODELS_DIR, f"lightgbm_fold{fold}.pkl"))
    print(f"   → modelo guardado: models/lightgbm_fold{fold}.pkl")

    # 5) OOF preds (para los índices va_idx relativos al train_base)
    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[va_idx] = val_pred

# OOF AUC (sobre train_base)
oof_auc = roc_auc_score(train_base_raw['isFraud'].values, oof_preds)
print(f"\nOOF AUC (train_base CV): {oof_auc:.6f}")

# -------------------------
# 9) Guardar IsolationForest (entrenado en train_base_enc)
# -------------------------
print("Entrenando IsolationForest en train_base_enc (para detección de anomalies)...")
# Prepara datos para isoforest: train_base_enc (ya tiene encoder_full+pca_full columns? we have train_base_enc above)
# But ensure train_base_enc has features columns
train_base_proc = train_base_enc.copy()
for f in features:
    if f not in train_base_proc.columns:
        train_base_proc[f] = 0.0
X_iso_train = train_base_proc[features].fillna(0)
iso_model, iso_scores = train_isolation_forest(X_iso_train, contamination=0.01)
joblib.dump(iso_model, os.path.join(MODELS_DIR, "isoforest.joblib"))
joblib.dump(features, os.path.join(MODELS_DIR, "isoforest_features.pkl"))
print("   → IsolationForest guardado y features list guardada.")

# -------------------------
# 10) Evaluación en Holdout (usando encoder_full + pca_full + selected_features)
# -------------------------
print("Evaluando en holdout (no visto durante training)...")
# aplicar encoder_full + pca_full a holdout_raw (ya lo hicimos arriba -> holdout_enc)
for f in features:
    if f not in holdout_enc.columns:
        holdout_enc[f] = 0.0
X_hold = holdout_enc[features].fillna(0)
y_hold = holdout_enc['isFraud'].values

# promedio de modelos
preds_hold = np.mean([m.predict(X_hold, num_iteration=m.best_iteration) for m in models], axis=0)
val_auc = roc_auc_score(y_hold, preds_hold)
print(f"HOLDOUT AUC: {val_auc:.6f}")

# -------------------------
# 11) Calcular threshold óptimo usando OOF (no holdout)
# -------------------------
precisions, recalls, thresholds = precision_recall_curve(train_base_raw['isFraud'].values, oof_preds)
f1s = 2*(precisions*recalls)/(precisions+recalls+1e-9)
best_i = np.nanargmax(f1s)
best_threshold = float(thresholds[best_i])
best_f1 = float(f1s[best_i])
print(f"Best threshold (from OOF) = {best_threshold:.4f} (F1={best_f1:.4f})")

# Guardar threshold y métricas
joblib.dump({"threshold": best_threshold, "best_f1": best_f1}, os.path.join(MODELS_DIR, "decision_threshold.pkl"))
metrics = {
    "oof_auc": float(oof_auc),
    "val_auc": float(val_auc),
    "best_threshold": best_threshold,
    "best_f1": best_f1
}
with open(os.path.join(OUTPUTS_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Entrenamiento completo. Artifacts guardados en models/ y outputs/")
