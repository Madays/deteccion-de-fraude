import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np
from src import preprocess, feature_engineering, utils
from glob import glob

# === 1. Cargar nuevas transacciones ===
new_data = pd.read_csv("data/new_transactions.csv")

# === 2. Preprocesar igual que en training ===
print("[1/4] Preprocesando...")
new_data = preprocess.basic_clean(new_data)
new_data = preprocess.basic_impute(new_data)

# Load encoder and get the fitted categorical columns
encoder = joblib.load("models/target_encoder.joblib")
cat_cols = encoder.cols  # List of columns the encoder was fitted on
# Filter to only those present in new_data
cat_cols = [c for c in cat_cols if c in new_data.columns]
# Apply encoding
new_data = preprocess.apply_target_encoding(new_data, encoder, cat_cols)
# Load encoder and define categorical columns (same as in training)



# === 3. Feature engineering ===
print("[2/4] Generando features...")
new_data = feature_engineering.log_transform_amount(new_data)
new_data = feature_engineering.create_time_features(new_data)
new_data = feature_engineering.card_agg_features(new_data)
new_data = feature_engineering.reduce_V_features_pca(new_data)

# Seleccionar las mismas columnas que en entrenamiento
features = joblib.load("models/selected_features.pkl")
X_new = new_data[features]

# === 4. Cargar modelos entrenados ===
print("[3/4] Cargando modelos...")
lgb_models = [joblib.load(m) for m in sorted(glob("models/lightgbm_fold*.pkl"))]
iso_model = joblib.load("models/isoforest.joblib")

# === 5. Predicciones LightGBM ===
print("[4/4] Prediciendo...")
lgb_preds = np.mean([m.predict(X_new) for m in lgb_models], axis=0)

# === 6. Predicciones Anomaly (Isolation Forest) ===
iso_scores = iso_model.decision_function(X_new)  # valores bajos = más anómalo
iso_anomalies = iso_model.predict(X_new)         # -1 = anomalía, 1 = normal

# === 7. Combinar o aplicar umbrales ===
# ejemplo: marcar como fraude si LGBM > 0.5 o es anomalía severa
fraud_flags = (lgb_preds > 0.5) | (iso_scores < -0.2)

# === 8. Guardar resultados ===
results = new_data.copy()
results["fraud_probability"] = lgb_preds
results["anomaly_score"] = iso_scores
results["is_fraud_pred"] = fraud_flags.astype(int)

results.to_csv("outputs/new_predictions.csv", index=False)
print("✅ Predicciones guardadas en outputs/new_predictions.csv")
