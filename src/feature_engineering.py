import pandas as pd
import numpy as np
import os
import joblib
from sklearn.decomposition import PCA

def log_transform_amount(df, col='TransactionAmt'):
    df = df.copy()
    if col in df.columns:
        new = pd.DataFrame({'TransactionAmt_log': np.log1p(df[col].astype(float))}, index=df.index)
        df = pd.concat([df, new], axis=1)
    return df

def create_time_features(df, dt_col='TransactionDT'):
    df = df.copy()
    if dt_col in df.columns and np.issubdtype(df[dt_col].dtype, np.number):
        new = pd.DataFrame({
            'hour': (df[dt_col] // 3600) % 24,
            'day': (df[dt_col] // (3600*24)) % 365
        }, index=df.index)
        df = pd.concat([df, new], axis=1)
    return df

def card_agg_features(df):
    df = df.copy()
    if 'card1' in df.columns and 'TransactionAmt' in df.columns:
        grp = df.groupby('card1')['TransactionAmt'].agg(['mean','std']).rename(columns={'mean':'card1_amt_mean','std':'card1_amt_std'})
        df = df.merge(grp, how='left', on='card1')
        new = pd.DataFrame({
            'TransactionAmt_to_card1_mean': df['TransactionAmt'] / (df['card1_amt_mean'] + 1e-9)
        }, index=df.index)
        df = pd.concat([df, new], axis=1)
    return df

def reduce_V_features_pca(df, n_components=20, training=True, model_path="models/pca_full.joblib"):
    """
    Aplica PCA sobre las columnas que empiezan en 'V'. 
    Si training=True -> entrena PCA (n_components=min(n_components, nV)) y lo guarda.
    Si training=False -> carga PCA desde model_path y transforma mismas columnas.
    Guarda/espera (pca, V_cols)
    """
    df = df.copy()
    V_cols = [c for c in df.columns if c.startswith('V') and df[c].dtype != 'object']
    if len(V_cols) < 5:
        return df

    if training:
        df_V = df[V_cols].fillna(-1)
        n_comp = min(n_components, len(V_cols))
        pca = PCA(n_components=n_comp, random_state=42)
        comp = pca.fit_transform(df_V)
        # Guardar pca y V_cols
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump((pca, V_cols), model_path)
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PCA model not found at {model_path}")
        pca, V_cols_train = joblib.load(model_path)
        # Asegurar que df tenga todas las columnas V usadas en entrenamiento
        miss = [c for c in V_cols_train if c not in df.columns]
        if miss:
            fill = pd.DataFrame({c: -1 for c in miss}, index=df.index)
            df = pd.concat([df, fill], axis=1)
        df_V = df[V_cols_train].fillna(-1)
        comp = pca.transform(df_V)

    pca_df = pd.DataFrame(comp, columns=[f'V_pca_{i}' for i in range(comp.shape[1])], index=df.index)
    df = pd.concat([df, pca_df], axis=1)
    return df

def select_features(df, max_features=100):
    """
    Selección simple: tomar features numéricas ordenadas por varianza (descendente),
    excluyendo IDs y target.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['TransactionID', 'isFraud']]
    # ordenar por varianza
    variances = df[num_cols].var().sort_values(ascending=False)
    selected = variances.index.tolist()[:max_features]
    return selected
