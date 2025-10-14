import os
import json
import joblib
import numpy as np
import pandas as pd
from glob import glob
from src import preprocess, feature_engineering

DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 1) Cargar nuevas transacciones
print("[1/6] Cargando nuevas transacciones...")
new_path = os.path.join(DATA_DIR, "new_transactions.csv")
if not os.path.exists(new_path):
    raise FileNotFoundError(f"No se encontró {new_path}")
new_df = pd.read_csv(new_path)
print(f"   → {len(new_df)} filas")

# 2) Preprocesamiento básico
print("[2/6] Preprocesando...")
new_df = preprocess.basic_clean(new_df)
new_df = preprocess.basic_impute(new_df)

# 3) Cargar encoder_full y aplicar target encoding
print("[3/6] Aplicando target encoding (encoder_full)...")
encoder_full = joblib.load(os.path.join(MODELS_DIR, "target_encoder_full.joblib"))
cat_cols = getattr(encoder_full, "cols", [])
# create missing cat cols in one shot
missing_cat = [c for c in cat_cols if c not in new_df.columns]
if missing_cat:
    fill = pd.DataFrame({c: 'missing' for c in missing_cat}, index=new_df.index)
    new_df = pd.concat([new_df, fill], axis=1)
new_df = preprocess.apply_target_encoding(new_df, encoder_full, cat_cols)

# 4) Feature engineering + PCA_full
print("[4/6] Generando features y aplicando PCA_full...")
new_df = feature_engineering.log_transform_amount(new_df)
new_df = feature_engineering.create_time_features(new_df)
new_df = feature_engineering.card_agg_features(new_df)
new_df = feature_engineering.reduce_V_features_pca(new_df, n_components=20, training=False, model_path=os.path.join(MODELS_DIR, "pca_full.joblib"))

# 5) Seleccionar features guardadas y alinear columnas
print("[5/6] Seleccionando y alineando features...")
features = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))
missing_feat = [c for c in features if c not in new_df.columns]
if missing_feat:
    fill = pd.DataFrame({c: 0 for c in missing_feat}, index=new_df.index)
    new_df = pd.concat([new_df, fill], axis=1)
X_new = new_df[features].fillna(0)

# 6) Cargar modelos LGBM y umbral
print("[6/6] Cargando modelos y threshold...")
model_files = sorted(glob(os.path.join(MODELS_DIR, "lightgbm_fold*.pkl")))
models = [joblib.load(m) for m in model_files]
th = joblib.load(os.path.join(MODELS_DIR, "decision_threshold.pkl"))['threshold']
print(f"   → Umbral cargado: {th:.4f}")

# 7) Predicciones LGBM
preds = np.mean([m.predict(X_new, num_iteration=m.best_iteration) for m in models], axis=0)

# 8) IsolationForest (opcional)
iso_path = os.path.join(MODELS_DIR, "isoforest.joblib")
iso_features_path = os.path.join(MODELS_DIR, "isoforest_features.pkl")
if os.path.exists(iso_path) and os.path.exists(iso_features_path):
    iso_model = joblib.load(iso_path)
    iso_feats = joblib.load(iso_features_path)
    # Alinear
    for c in iso_feats:
        if c not in X_new.columns:
            X_new[c] = 0
    X_iso = X_new[iso_feats]
    iso_scores = iso_model.decision_function(X_iso)
else:
    iso_scores = np.zeros(len(X_new))

# 9) Decisión final
fraud_flags = (preds >= th) | (iso_scores < -0.2)

# 10) Guardar resultados y resumen
os.makedirs(OUTPUTS_DIR, exist_ok=True)
out = new_df.copy()
out['fraud_probability'] = preds
out['anomaly_score'] = iso_scores
out['is_fraud_pred'] = fraud_flags.astype(int)
out.to_csv(os.path.join(OUTPUTS_DIR, "new_predictions.csv"), index=False)

# summary
total = len(out)
detected = int(out['is_fraud_pred'].sum())
pct = detected/total*100 if total>0 else 0
print(f"✅ Guardadas predicciones: {os.path.join(OUTPUTS_DIR, 'new_predictions.csv')}")
print(f"   → {detected} transacciones marcadas como fraude ({pct:.2f}%)")
print("Top 10 por probabilidad:")
print(out.sort_values("fraud_probability", ascending=False).head(10)[['TransactionID','fraud_probability','anomaly_score','is_fraud_pred']])
