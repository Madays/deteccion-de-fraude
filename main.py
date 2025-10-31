#!/usr/bin/env python3
# train_fraud_model.py
"""
Pipeline completo (versión profesional) para entrenar un modelo de detección de fraude
con soporte de features espaciales (latitude/longitude) usando el dataset:
    transaction_data.csv

Mejoras incluidas:
 - Calibración isotónica usando CalibratedClassifierCV(estimator=...)
 - Oversampling de positivos (en lugar de undersampling)
 - Features espaciales: distance_from_last_location, log_distance, speed_kmh
 - Mejoras de hiperparámetros LightGBM: num_leaves reducido, learning_rate aumentado
 - HTML report (simple) con métricas y gráficos (matplotlib + pandas)
 - Logs, tiempos por módulo, guardado de artefactos (models, scalers, encoders, metrics)

Modo ejecución local (CPU).
Dependencias: pandas, numpy, scikit-learn, lightgbm, joblib, matplotlib, category_encoders (opcional)
Instalación ejemplo:
    pip install pandas numpy scikit-learn lightgbm joblib matplotlib category_encoders
"""

import os
import time
import json
import logging
from functools import wraps
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, roc_curve, auc)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import catboost as cb
import optuna

import joblib
import matplotlib.pyplot as plt

# Optional: TargetEncoder from category_encoders (recommended for high-cardinality categoricals)
try:
    from category_encoders import TargetEncoder
except Exception:
    TargetEncoder = None

# LightGBM sklearn API (CPU)
try:
    import lightgbm as lgb
    from lightgbm import early_stopping
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# -------------------------
# Configuration
# -------------------------
RANDOM_STATE = 42
DATA_PATH = "transaction_data.csv"        # Cambia si tu archivo tiene otro nombre
OUTPUT_DIR = "outputs_prototype"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.html")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging (console + file)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "train_fraud_model.log"))
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# -------------------------
# Helpers & utilities
# -------------------------
def timed(func):
    """Decorator to time functions and log duration."""
    @wraps(func)
    def wrapper(*a, **kw):
        logger.info("START  - %s", func.__name__)
        t0 = time.time()
        res = func(*a, **kw)
        elapsed = time.time() - t0
        logger.info("END    - %s (%.2fs)", func.__name__, elapsed)
        return res
    return wrapper

def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return first matching column name among candidates (case-insensitive), or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    """Compute haversine distance in kilometers; returns np.nan on invalid inputs."""
    try:
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return np.nan
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0
        return R * c
    except Exception:
        return np.nan

# -------------------------
# 1. Load & EDA
# -------------------------
@timed
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV, normalize column names, detect target/timestamp and print basic EDA logs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    logger.info("Loaded data shape: %s", df.shape)

    # Detect target
    target_candidates = [c for c in df.columns if c.lower() in ("fraud_flag", "fraudflag", "fraud", "isfraud")]
    if not target_candidates:
        raise ValueError("No target column found. Expected 'Fraud Flag' or similar.")
    target_col = target_candidates[0]
    df.rename(columns={target_col: "Fraud_Flag"}, inplace=True)
    df["Fraud_Flag"] = df["Fraud_Flag"].astype(int)

    # Detect timestamp
    ts_candidates = [c for c in df.columns if c.lower() in ("timestamp", "transactiondt", "date", "datetime")]
    if ts_candidates:
        ts = ts_candidates[0]
        try:
            df["Timestamp"] = pd.to_datetime(df[ts])
        except Exception:
            try:
                df["Timestamp"] = pd.to_datetime(df[ts].astype(float), unit="s")
            except Exception:
                logger.warning("Could not parse timestamp column %s to datetime.", ts)
    else:
        logger.warning("No timestamp column found; pipeline will fallback to random split.")

    # Quick EDA
    logger.info("Top null counts:\n%s", df.isnull().sum().sort_values(ascending=False).head(30).to_string())
    logger.info("Target distribution: positives=%d, negatives=%d, pos_rate=%.6f",
                int(df["Fraud_Flag"].sum()), int(len(df) - df["Fraud_Flag"].sum()), df["Fraud_Flag"].mean())
    return df

# -------------------------
# 2. Basic preprocess + parse geolocation
# -------------------------
@timed
def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, strip strings, minimal cleaning."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows", before - len(df))
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

@timed
def parse_geolocation(df: pd.DataFrame, candidates: List[str] = None) -> pd.DataFrame:
    """
    Parse geolocation column containing "lat N, lon W" or similar into latitude and longitude
    Adds columns 'latitude' and 'longitude' (floats).
    """
    df = df.copy()
    if candidates is None:
        candidates = ["Geolocation (Latitude/Longitude)", "Geolocation", "Geolocation_Latitude_Longitude", "Geolocation_(Latitude/Longitude)"]
    col = find_col(df, candidates)
    if col is None:
        logger.warning("No geolocation column found among candidates; adding empty latitude/longitude.")
        df["latitude"] = np.nan
        df["longitude"] = np.nan
        return df

    s = df[col].astype(str).fillna("")
    # Split by comma, then extract numbers
    parts = s.str.split(",", expand=True)
    if parts.shape[1] < 2:
        logger.warning("Geolocation column found but could not split reliably; adding NaNs.")
        df["latitude"] = np.nan
        df["longitude"] = np.nan
        return df
    # Extract numeric part from first part (e.g., "34.0522 N" -> 34.0522)
    lat_str = parts[0].str.extract(r'([+-]?\d*\.?\d+)')[0]
    lon_str = parts[1].str.extract(r'([+-]?\d*\.?\d+)')[0]
    df["latitude"] = pd.to_numeric(lat_str, errors="coerce")
    df["longitude"] = pd.to_numeric(lon_str, errors="coerce")
    logger.info("Parsed geolocation to latitude/longitude; missing lat/lon: %d", df[["latitude", "longitude"]].isnull().any(axis=1).sum())
    return df

# -------------------------
# 3. Feature engineering
# -------------------------
@timed
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal, account, transactional, device/network and spatial features.
    Adds:
      - hour, dayofweek, is_weekend
      - time_since_last_txn_user, txn_count_last_5min (approx)
      - sender/receiver aggregates
      - amount_zscore_user, is_high_value, same_receiver_amount_count
      - device_freq, avg_latency_user, bandwidth_ratio
      - distance_from_last_location, log_distance, speed_kmh, avg_txn_distance, suspicious_travel
    """
    df = df.copy()

    # heuristics for column names
    amt_col = find_col(df, ["Transaction Amount", "TransactionAmount", "Amount", "Transaction_Amount_(USD)"])
    sender_col = find_col(df, ["Sender Account ID", "SenderID", "from_account", "sender_id"])
    receiver_col = find_col(df, ["Receiver Account ID", "ReceiverID", "to_account", "receiver_id"])
    ts_col = find_col(df, ["Timestamp", "TransactionDT", "timestamp", "Date", "datetime"])
    device_col = find_col(df, ["Device Used", "Device", "device"])
    latency_col = find_col(df, ["Latency (ms)","Latency", "Latency_ms", "Latency_(ms)"])
    bandwidth_col = find_col(df, ["Slice Bandwidth (Mbps)", "Slice_Bandwidth", "Slice_Bandwidth_Mbps", "Bandwidth"])
    # parse geolocation
    df = parse_geolocation(df)

    # temporal features
    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["hour"] = df[ts_col].dt.hour.fillna(0).astype(int)
        df["dayofweek"] = df[ts_col].dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    else:
        df["hour"] = 0
        df["dayofweek"] = 0
        df["is_weekend"] = 0

    # time deltas per sender
    if sender_col and sender_col in df.columns and ts_col and ts_col in df.columns:
        df = df.sort_values([sender_col, ts_col]).reset_index(drop=True)
        df["time_since_last_txn_user"] = df.groupby(sender_col)[ts_col].diff().dt.total_seconds().fillna(1e9)
        # txn_count_last_5min: rolling counts (approx) - may be heavy; fallback to cumcount
        try:
            df["_ones"] = 1
            df["txn_count_last_5min"] = df.groupby(sender_col).rolling("5min", on=ts_col)["_ones"].count().values
            df.drop(columns=["_ones"], inplace=True)
        except Exception:
            df["txn_count_last_5min"] = df.groupby(sender_col).cumcount().clip(upper=50)
    else:
        df["time_since_last_txn_user"] = 1e9
        df["txn_count_last_5min"] = 0

    # per-account aggregates
    if sender_col and amt_col and sender_col in df.columns and amt_col in df.columns:
        grp_s = df.groupby(sender_col)[amt_col].agg(["count", "mean", "std"]).rename(columns={"count":"sender_txn_count","mean":"sender_avg_amount","std":"sender_std_amount"})
        df = df.merge(grp_s, how="left", left_on=sender_col, right_index=True)
    else:
        df["sender_txn_count"] = 0
        df["sender_avg_amount"] = 0.0
        df["sender_std_amount"] = 1.0

    if receiver_col and amt_col and receiver_col in df.columns and amt_col in df.columns:
        grp_r = df.groupby(receiver_col)[amt_col].agg(["count", "mean"]).rename(columns={"count":"receiver_txn_count","mean":"receiver_avg_amount"})
        df = df.merge(grp_r, how="left", left_on=receiver_col, right_index=True)
    else:
        df["receiver_txn_count"] = 0
        df["receiver_avg_amount"] = 0.0

    # transactional features
    if amt_col and sender_col and amt_col in df.columns and sender_col in df.columns:
        df["sender_std_amount"] = df["sender_std_amount"].replace(0, 1.0)
        df["amount_zscore_user"] = (df[amt_col] - df["sender_avg_amount"]) / (df["sender_std_amount"] + 1e-9)
        df["is_high_value"] = (df[amt_col] >= df.groupby(sender_col)[amt_col].transform(lambda x: x.quantile(0.95))).astype(int)
    else:
        df["amount_zscore_user"] = 0.0
        df["is_high_value"] = 0

    if sender_col and receiver_col and amt_col and all(c in df.columns for c in [sender_col, receiver_col, amt_col]):
        df["same_receiver_amount_count"] = df.groupby([sender_col, receiver_col, amt_col])[amt_col].transform("count")
    else:
        df["same_receiver_amount_count"] = 0

    # device/network features
    if device_col and device_col in df.columns and sender_col and sender_col in df.columns:
        df["device_freq"] = df.groupby(device_col)[sender_col].transform("nunique").fillna(0)
    else:
        df["device_freq"] = 0

    if latency_col and latency_col in df.columns:
        df[latency_col] = pd.to_numeric(df[latency_col], errors="coerce").fillna(0)
        if sender_col in df.columns:
            df["avg_latency_user"] = df.groupby(sender_col)[latency_col].transform("mean").fillna(0)
        else:
            df["avg_latency_user"] = df[latency_col]
    else:
        df["avg_latency_user"] = 0.0

    if bandwidth_col and bandwidth_col in df.columns and latency_col and latency_col in df.columns:
        df[bandwidth_col] = pd.to_numeric(df[bandwidth_col], errors="coerce").fillna(0.0)
        df["bandwidth_ratio"] = df[bandwidth_col] / (df[latency_col] + 1e-6)
    else:
        df["bandwidth_ratio"] = 0.0

    # spatial features: consecutive distance, log-distance and speed (km/h)
    if 'latitude' in df.columns and 'longitude' in df.columns and sender_col and sender_col in df.columns and ts_col and ts_col in df.columns:
        df = df.sort_values([sender_col, ts_col]).reset_index(drop=True)
        df["prev_lat"] = df.groupby(sender_col)["latitude"].shift(1)
        df["prev_lon"] = df.groupby(sender_col)["longitude"].shift(1)
        df["distance_from_last_location"] = df.apply(
            lambda r: haversine_km(r["prev_lat"], r["prev_lon"], r["latitude"], r["longitude"]), axis=1
        ).fillna(0.0)
        # log distance to reduce skew
        df["log_distance"] = np.log1p(df["distance_from_last_location"])
        # speed km/h: distance (km) / (time hours)
        # time_since_last_txn_user is in seconds -> convert to hours
        df["time_since_last_txn_user_hr"] = df["time_since_last_txn_user"] / 3600.0
        df["speed_kmh"] = df["distance_from_last_location"] / (df["time_since_last_txn_user_hr"] + 1e-9)
        df["speed_kmh"] = df["speed_kmh"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["avg_txn_distance"] = df.groupby(sender_col)["distance_from_last_location"].transform("mean").fillna(0.0)
        # suspicious travel if very far and very quick
        df["suspicious_travel"] = ((df["distance_from_last_location"] > 500) & (df["time_since_last_txn_user"] < 1800)).astype(int)
    else:
        df["distance_from_last_location"] = 0.0
        df["log_distance"] = 0.0
        df["time_since_last_txn_user_hr"] = 1e9
        df["speed_kmh"] = 0.0
        df["avg_txn_distance"] = 0.0
        df["suspicious_travel"] = 0

    # pin/security features (if present)
    if "PIN_Code" in df.columns:
        def pin_entropy(pin):
            try:
                s = str(pin)
                counts = np.array([s.count(ch) for ch in set(s)], dtype=float)
                probs = counts / counts.sum()
                ent = -(probs * np.log2(probs + 1e-9)).sum()
                return float(ent)
            except Exception:
                return 0.0
        df["pin_entropy"] = df["PIN_Code"].apply(pin_entropy)
        df["pin_reuse_count"] = df.groupby("PIN_Code")["PIN_Code"].transform("count")
        if device_col and device_col in df.columns and sender_col and sender_col in df.columns:
            df["pin_device_match"] = df.groupby(["Sender_Account_ID", "PIN_Code"])[device_col].transform("nunique").fillna(0)
        else:
            df["pin_device_match"] = 0
    else:
        df["pin_entropy"] = 0.0
        df["pin_reuse_count"] = 0
        df["pin_device_match"] = 0

    # final numeric cleaning
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # store detected names as attributes
    df.attrs["__amt_col__"] = amt_col
    df.attrs["__sender_col__"] = sender_col
    df.attrs["__receiver_col__"] = receiver_col
    df.attrs["__device_col__"] = device_col
    df.attrs["__latency_col__"] = latency_col
    df.attrs["__bandwidth_col__"] = bandwidth_col
    return df

# -------------------------
# 4. Splits and sampling
# -------------------------
@timed
def time_based_split(df: pd.DataFrame, test_size: float = 0.2, ts_col: str = "Timestamp") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: train = earliest (1-test_size), val = latest test_size. Falls back to stratified random if no timestamp."""
    if ts_col in df.columns:
        df_sorted = df.sort_values(ts_col).reset_index(drop=True)
        cutoff = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:cutoff].reset_index(drop=True)
        val_df = df_sorted.iloc[cutoff:].reset_index(drop=True)
        logger.info("Time split: train %d rows, val %d rows", len(train_df), len(val_df))
    else:
        logger.warning("Timestamp missing: falling back to random stratified split.")
        train_df, val_df = train_test_split(df, test_size=test_size, stratify=df["Fraud_Flag"], random_state=RANDOM_STATE)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
    return train_df, val_df

@timed
def oversample_train(train_df: pd.DataFrame, target_col: str = "Fraud_Flag", neg_pos_ratio: int = 5) -> pd.DataFrame:
    """
    Use SMOTE to oversample positives so that negative:positive ratio is approximately neg_pos_ratio.
    Keeps all negatives, generates synthetic positives using SMOTE.
    """
    pos = train_df[train_df[target_col] == 1]
    neg = train_df[train_df[target_col] == 0]
    n_neg = len(neg)
    if len(pos) == 0:
        raise ValueError("No positive examples in training split.")
    # desired number of positives so that neg / pos ~= neg_pos_ratio
    desired_pos = int(n_neg // max(1, neg_pos_ratio))
    desired_pos = max(desired_pos, len(pos))  # never downsample positives here

    if desired_pos > len(pos):
        # Use SMOTE to generate synthetic samples
        smote = SMOTE(sampling_strategy={1: desired_pos}, random_state=RANDOM_STATE, k_neighbors=min(5, len(pos)-1))
        # Need to separate features and target
        X = train_df.drop(columns=[target_col, "Timestamp"])  # exclude target and timestamp
        y = train_df[target_col]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        # Reconstruct DataFrame
        train_bal = pd.concat([X_resampled, y_resampled], axis=1)
        train_bal = train_bal.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        logger.info("SMOTE oversampled train: negatives=%d, positives(original)=%d, positives(after SMOTE)=%d", n_neg, len(pos), desired_pos)
    else:
        train_bal = train_df
        logger.info("No oversampling needed: negatives=%d, positives=%d", n_neg, len(pos))
    return train_bal

# -------------------------
# 5. Encoding & scaling
# -------------------------
@timed
def encode_and_scale(train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str = "Fraud_Flag", onehot_thresh: int = 10):
    """
    One-hot low-cardinality categoricals; TargetEncoder (if available) for high-cardinality.
    Fit encoders & scaler on train, transform both train & val.
    Returns X_train, X_val, y_train, y_val, metadata.
    """
    metadata = {}
    # categorical candidates
    cat_candidates = [c for c in train_df.columns if train_df[c].dtype == "object" or train_df[c].dtype.name == "category"]
    cat_candidates = [c for c in cat_candidates if c not in ("Timestamp", target_col)]
    low_card = [c for c in cat_candidates if train_df[c].nunique() <= onehot_thresh]
    high_card = [c for c in cat_candidates if train_df[c].nunique() > onehot_thresh]
    logger.info("Low-cardinal cats: %s", low_card)
    logger.info("High-cardinal cats: %s", high_card)

    train_tmp = train_df.copy()
    val_tmp = val_df.copy()

    # One-hot low-card by concatenating to ensure same columns
    if low_card:
        combined = pd.concat([train_tmp, val_tmp], axis=0, ignore_index=True)
        combined = pd.get_dummies(combined, columns=low_card, dummy_na=False, drop_first=False)
        train_tmp = combined.iloc[:len(train_tmp)].reset_index(drop=True)
        val_tmp = combined.iloc[len(train_tmp):].reset_index(drop=True)

    # Target encoding high-card: fit on train only; fallback to LabelEncoder if category_encoders missing
    encoders = {}
    if high_card:
        if TargetEncoder is None:
            logger.warning("category_encoders not found; using LabelEncoder fallback for high-card.")
            for c in high_card:
                le = LabelEncoder()
                train_tmp[c] = train_tmp[c].astype(str).fillna("missing")
                val_tmp[c] = val_tmp[c].astype(str).fillna("missing")
                # fit on combined unique to avoid unseen
                classes = pd.concat([train_tmp[c], val_tmp[c]], axis=0).unique()
                le.fit(classes)
                train_tmp[c] = le.transform(train_tmp[c])
                val_tmp[c] = val_tmp[c].map(lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1)
                encoders[c] = ("label", le)
        else:
            te = TargetEncoder(cols=high_card, smoothing=0.3)
            te.fit(train_tmp[high_card], train_tmp[target_col])
            train_tmp[high_card] = te.transform(train_tmp[high_card])
            val_tmp[high_card] = te.transform(val_tmp[high_card])
            encoders["target_encoder"] = te

    # Build feature list excluding target & timestamp & ID-like columns
    drop_cols = [target_col, "Timestamp"]
    feature_cols = [c for c in train_tmp.columns if c not in drop_cols]
    for id_cand in ["TransactionID", "Transaction_ID", "Transaction_Id", "ID"]:
        if id_cand in feature_cols:
            feature_cols.remove(id_cand)

    numeric_feat = [c for c in feature_cols if train_tmp[c].dtype != "object"]

    scaler = RobustScaler()
    scaler.fit(train_tmp[numeric_feat].fillna(0.0))
    train_tmp[numeric_feat] = scaler.transform(train_tmp[numeric_feat].fillna(0.0))
    val_tmp[numeric_feat] = scaler.transform(val_tmp[numeric_feat].fillna(0.0))

    X_train = train_tmp[feature_cols].copy()
    X_val = val_tmp[feature_cols].copy()
    y_train = train_tmp[target_col].astype(int).copy()
    y_val = val_tmp[target_col].astype(int).copy()

    metadata["feature_cols"] = feature_cols
    metadata["numeric_feat"] = numeric_feat
    metadata["scaler"] = scaler
    metadata["encoders"] = encoders
    return X_train, X_val, y_train, y_val, metadata

# -------------------------
# 6. Model training (LightGBM) - adjusted hyperparams
# -------------------------
@timed
def train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight: float, early_stopping_rounds: int = 100):
    """
    Train sklearn LGBMClassifier with tuned hyperparams for prototyping.
    Reduced num_leaves and higher learning_rate to reduce overfitting & speed up.
    """
    if not LGB_AVAILABLE:
        raise RuntimeError("LightGBM not installed. Install with pip install lightgbm")

    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.03,    # slower learning for better generalization
        num_leaves=64,         # increased for more complexity
        n_estimators=3000,     # more trees
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,         # increased regularization
        reg_lambda=0.2,
        min_child_samples=20,  # smaller for more splits
        max_depth=8,           # limit depth to prevent overfitting
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(early_stopping_rounds)]
    )
    return clf

# -------------------------
# 7. Thresholds & evaluation helpers
# -------------------------
def youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    idx = int(np.nanargmax(J))
    return float(thresholds[idx])

def best_threshold_by_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    p, r, thr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (p * r) / (p + r + 1e-12)
    if len(f1) == 0:
        return 0.5, 0.0
    idx = int(np.nanargmax(f1))
    if idx >= len(thr):
        best_thr = float(thr[-1]) if len(thr) > 0 else 0.5
    else:
        best_thr = float(thr[idx])
    return best_thr, float(f1[idx])

def evaluate_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_scores >= threshold).astype(int)
    auc_score = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float("nan")
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-12)
    return {"auc": float(auc_score), "precision": float(prec), "recall": float(rec),
            "f1": float(f1v), "fpr": float(fpr), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

# -------------------------
# 8. Plots + HTML report helpers
# -------------------------
@timed
def plot_curves(y_val, probs, prefix=PLOTS_DIR):
    os.makedirs(prefix, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(prefix, "roc_curve.png")); plt.close()

    # PR Precision-Recall
    precision, recall, _ = precision_recall_curve(y_val, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"PR (AUC={pr_auc:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(prefix, "pr_curve.png")); plt.close()

@timed
def plot_feature_importance(model, feature_names, outpath=os.path.join(PLOTS_DIR, "feature_importance.png"), top_n=40):
    try:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        names = [feature_names[i] for i in idx]
        vals = importances[idx]
        plt.figure(figsize=(8, min(12, top_n/2)))
        plt.barh(range(len(vals))[::-1], vals[::-1], align="center")
        plt.yticks(range(len(vals))[::-1], names[::-1])
        plt.xlabel("Feature importance"); plt.title("Top feature importances"); plt.tight_layout()
        plt.savefig(outpath); plt.close()
    except Exception as e:
        logger.warning("Could not plot feature importance: %s", e)

@timed
def write_html_report(metrics: Dict[str, Any], plots: List[str], outpath: str = REPORT_PATH):
    """
    Simple HTML report that embeds saved plot images and prints metrics table.
    Uses only local images and inline HTML (no external libs).
    """
    html_parts = []
    html_parts.append("<html><head><meta charset='utf-8'><title>Fraud Model Report</title></head><body>")
    html_parts.append("<h1>Fraud Detection Model Report</h1>")
    html_parts.append("<h2>Metrics</h2>")
    html_parts.append("<pre>{}</pre>".format(json.dumps(metrics, indent=2)))
    html_parts.append("<h2>Plots</h2>")
    for p in plots:
        if os.path.exists(p):
            html_parts.append(f"<div><img src='{os.path.basename(p)}' style='max-width:900px'></div><br>")
    # Save html and copy images to same folder
    report_dir = os.path.dirname(outpath)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    # Copy images to report dir (already in PLOTS_DIR), but ensure paths are relative
    # (we embed only basename; ensure report opened from OUTPUT_DIR)
    logger.info("Wrote HTML report to %s (images must be served from %s)", outpath, PLOTS_DIR)

# -------------------------
# 9. Main pipeline
# -------------------------
@timed
def run_pipeline(data_path: str = DATA_PATH,
                 neg_pos_ratio: int = 5,
                 calibrate: bool = True,
                 iso_contamination: float = 0.01,
                 ensemble_alpha: float = 0.85):
    # Load
    df = load_data(data_path)

    # Basic preprocess
    df = preprocess_basic(df)

    # Feature engineering
    df = feature_engineering(df)

    # Time split
    train_full, val = time_based_split(df, test_size=0.2, ts_col="Timestamp")

    # Oversample positives (instead of undersampling) to target neg_pos_ratio
    train = oversample_train(train_full, target_col="Fraud_Flag", neg_pos_ratio=neg_pos_ratio)

    # Encode & scale (fit on train)
    X_train, X_val, y_train, y_val, metadata = encode_and_scale(train, val, target_col="Fraud_Flag", onehot_thresh=10)
    feature_cols = metadata["feature_cols"]
    numeric_feat = metadata["numeric_feat"]
    logger.info("Using %d features", len(feature_cols))

    # IsolationForest trained on numeric features (train) -> anomaly score on val
    iso = IsolationForest(n_estimators=200, contamination=iso_contamination, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train[numeric_feat].fillna(-1))
    joblib.dump(iso, os.path.join(MODEL_DIR, "isoforest.joblib"))
    iso_scores_val = -iso.decision_function(X_val[numeric_feat].fillna(-1))  # higher = more anomalous
    # normalize iso score to [0,1]
    iso_min, iso_max = np.min(iso_scores_val), np.max(iso_scores_val)
    if iso_max - iso_min > 1e-9:
        iso_scaled = (iso_scores_val - iso_min) / (iso_max - iso_min)
    else:
        iso_scaled = np.zeros_like(iso_scores_val)

    # class weight / scale_pos_weight
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(1, pos))
    logger.info("scale_pos_weight: %.3f (pos=%d, neg=%d)", scale_pos_weight, pos, neg)

    # Train LGBM
    clf = train_lightgbm(X_train[feature_cols], y_train, X_val[feature_cols], y_val, scale_pos_weight=scale_pos_weight, early_stopping_rounds=100)
    joblib.dump(clf, os.path.join(MODEL_DIR, "lgbm_model.joblib"))

    # Cross-validation for better evaluation
    cv_scores = cross_val_score(clf, X_train[feature_cols], y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    logger.info("Cross-validation AUC scores: %s (mean=%.4f, std=%.4f)", cv_scores, cv_scores.mean(), cv_scores.std())

    # Train additional models for stacking
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train[feature_cols], y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))

    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_clf.fit(X_train[feature_cols], y_train)
    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "xgb_model.joblib"))

    cb_clf = cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=0)
    cb_clf.fit(X_train[feature_cols], y_train)
    joblib.dump(cb_clf, os.path.join(MODEL_DIR, "cb_model.joblib"))

    # Stacking ensemble
    estimators = [
        ('lgbm', clf),
        ('rf', rf),
        ('xgb', xgb_clf),
        ('cb', cb_clf)
    ]
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=lgb.LGBMClassifier(n_estimators=50, random_state=RANDOM_STATE), cv=3, n_jobs=-1)
    stacking_clf.fit(X_train[feature_cols], y_train)
    joblib.dump(stacking_clf, os.path.join(MODEL_DIR, "stacking_model.joblib"))

    # Predict on val
    if hasattr(clf, "predict_proba"):
        yval_proba = clf.predict_proba(X_val[feature_cols])[:, 1]
    else:
        yval_proba = clf.predict(X_val[feature_cols])

    # Get predictions from all models
    rf_proba = rf.predict_proba(X_val[feature_cols])[:, 1]
    xgb_proba = xgb_clf.predict_proba(X_val[feature_cols])[:, 1]
    cb_proba = cb_clf.predict_proba(X_val[feature_cols])[:, 1]
    stacking_proba = stacking_clf.predict_proba(X_val[feature_cols])[:, 1]

    # Ensemble LGBM + IsolationForest score
    ensembled_prob = ensemble_alpha * yval_proba + (1.0 - ensemble_alpha) * iso_scaled

    # Advanced ensemble: weighted average of all models
    advanced_ensemble = (0.4 * yval_proba + 0.2 * rf_proba + 0.2 * xgb_proba + 0.2 * cb_proba + 0.3 * stacking_proba + 0.1 * iso_scaled)

    # Choose thresholds for original ensemble
    thr_youden = youden_threshold(y_val.values, ensembled_prob)
    thr_f1, best_f1 = best_threshold_by_f1(y_val.values, ensembled_prob)
    metrics_youden = evaluate_at_threshold(y_val.values, ensembled_prob, thr_youden)
    metrics_f1 = evaluate_at_threshold(y_val.values, ensembled_prob, thr_f1)
    logger.info("Youden threshold %.4f => %s", thr_youden, json.dumps(metrics_youden))
    logger.info("F1 threshold %.4f (best f1=%.4f) => %s", thr_f1, best_f1, json.dumps(metrics_f1))

    # Evaluate advanced ensemble
    thr_youden_adv = youden_threshold(y_val.values, advanced_ensemble)
    thr_f1_adv, best_f1_adv = best_threshold_by_f1(y_val.values, advanced_ensemble)
    metrics_youden_adv = evaluate_at_threshold(y_val.values, advanced_ensemble, thr_youden_adv)
    metrics_f1_adv = evaluate_at_threshold(y_val.values, advanced_ensemble, thr_f1_adv)
    logger.info("Advanced Ensemble Youden threshold %.4f => %s", thr_youden_adv, json.dumps(metrics_youden_adv))
    logger.info("Advanced Ensemble F1 threshold %.4f (best f1=%.4f) => %s", thr_f1_adv, best_f1_adv, json.dumps(metrics_f1_adv))

    # Calibration: use CalibratedClassifierCV with estimator parameter correctly.
    calibrated_info = {}
    calibrated_probs = None
    if calibrate:
        try:
            # We create a fresh estimator with similar hyperparams to be used inside CalibratedClassifierCV.
            base_estimator = lgb.LGBMClassifier(
                objective="binary",
                learning_rate=0.05,
                num_leaves=48,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=30,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
                scale_pos_weight=scale_pos_weight
            )
            # CalibratedClassifierCV accepts parameter `estimator` in modern sklearn. We fit inside cv=3 (slower but safe).
            calibrator = CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=3)
            calibrator.fit(X_train[feature_cols], y_train)
            calibrated_probs = calibrator.predict_proba(X_val[feature_cols])[:, 1]
            # ensemble calibrated + iso
            cal_ensembled = ensemble_alpha * calibrated_probs + (1.0 - ensemble_alpha) * iso_scaled
            thr_cal = youden_threshold(y_val.values, cal_ensembled)
            metrics_cal = evaluate_at_threshold(y_val.values, cal_ensembled, thr_cal)
            calibrated_info = {"calibrator_saved": True, "thr_cal": thr_cal, "metrics_cal": metrics_cal}
            joblib.dump(calibrator, os.path.join(MODEL_DIR, "calibrator_isotonic.joblib"))
            logger.info("Calibration done. thr_cal %.4f -> %s", thr_cal, json.dumps(metrics_cal))
        except Exception as e:
            logger.exception("Calibration failed: %s", e)
            calibrated_info = {"calibrator_saved": False, "error": str(e)}

    # Final metrics / artifacts
    out = {
        "train_full_rows": int(len(train_full)),
        "train_used_rows": int(len(train)),
        "val_rows": int(len(val)),
        "scale_pos_weight": scale_pos_weight,
        "ensemble_alpha": ensemble_alpha,
        "youden_threshold": thr_youden,
        "youden_metrics": metrics_youden,
        "f1_threshold": thr_f1,
        "f1_metrics": metrics_f1,
        "best_f1_value": best_f1,
        "advanced_ensemble_youden_threshold": thr_youden_adv,
        "advanced_ensemble_youden_metrics": metrics_youden_adv,
        "advanced_ensemble_f1_threshold": thr_f1_adv,
        "advanced_ensemble_f1_metrics": metrics_f1_adv,
        "advanced_ensemble_best_f1_value": best_f1_adv,
        "calibrated_info": calibrated_info,
        "raw_auc_lgbm": float(roc_auc_score(y_val.values, yval_proba)) if len(np.unique(y_val.values))>1 else None,
        "raw_auc_rf": float(roc_auc_score(y_val.values, rf_proba)) if len(np.unique(y_val.values))>1 else None,
        "raw_auc_xgb": float(roc_auc_score(y_val.values, xgb_proba)) if len(np.unique(y_val.values))>1 else None,
        "raw_auc_cb": float(roc_auc_score(y_val.values, cb_proba)) if len(np.unique(y_val.values))>1 else None,
        "raw_auc_stacking": float(roc_auc_score(y_val.values, stacking_proba)) if len(np.unique(y_val.values))>1 else None,
        "raw_auc_advanced_ensemble": float(roc_auc_score(y_val.values, advanced_ensemble)) if len(np.unique(y_val.values))>1 else None
    }
    with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Save models / metadata
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
    joblib.dump(metadata.get("scaler"), os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(metadata.get("encoders"), os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "lgbm_model_raw.joblib"))

    # Plots + feature importance
    plot_curves(y_val.values, ensembled_prob, prefix=PLOTS_DIR)
    plot_feature_importance(clf, feature_cols, outpath=os.path.join(PLOTS_DIR, "feature_importance.png"))

    # Write a minimal HTML report (images referenced by basename; open from OUTPUT_DIR root)
    plots_list = [os.path.join(PLOTS_DIR, "roc_curve.png"), os.path.join(PLOTS_DIR, "pr_curve.png"), os.path.join(PLOTS_DIR, "feature_importance.png")]
    # Copy plot files to OUTPUT_DIR (report references basenames)
    for p in plots_list:
        if os.path.exists(p):
            dest = os.path.join(OUTPUT_DIR, os.path.basename(p))
            try:
                from shutil import copyfile
                copyfile(p, dest)
            except Exception:
                pass
    write_html_report(out, plots_list, outpath=REPORT_PATH)

    logger.info("Pipeline finished. Artifacts in %s", OUTPUT_DIR)
    return out

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    start = time.time()
    logger.info("Starting training pipeline (local CPU) with improved settings and geolocation features.")
    metrics = run_pipeline(data_path=DATA_PATH, neg_pos_ratio=5, calibrate=True, iso_contamination=0.01, ensemble_alpha=0.85)
    logger.info("Finished in %.2fs. Metrics:\n%s", time.time() - start, json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
