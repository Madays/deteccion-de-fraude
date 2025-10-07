import pandas as pd
import numpy as np

def log_transform_amount(df, col='TransactionAmt'):
    if col in df.columns:
        df['TransactionAmt_log'] = np.log1p(df[col].astype(float))
    return df

def create_time_features(df, dt_col='TransactionDT'):
    # TransactionDT in IEEE dataset is seconds since some point - create hour/day
    try:
        df['hour'] = (df[dt_col] // 3600) % 24
        df['day'] = (df[dt_col] // (3600*24)) % 365
    except Exception:
        # If TransactionDT not numeric, ignore
        pass
    return df

def card_agg_features(df):
    # Example: amount relative to card mean/std (group by card1 if present)
    if 'card1' in df.columns:
        grp = df.groupby('card1')['TransactionAmt'].agg(['mean','std']).rename(columns={'mean':'card1_amt_mean','std':'card1_amt_std'})
        df = df.merge(grp, how='left', on='card1')
        df['TransactionAmt_to_card1_mean'] = df['TransactionAmt'] / (df['card1_amt_mean'] + 1e-6)
    return df

def reduce_V_features_pca(df, n_components=20):
    from sklearn.decomposition import PCA
    V_cols = [c for c in df.columns if c.startswith('V') and df[c].dtype != 'object']
    if len(V_cols) >= 5:
        df_V = df[V_cols].fillna(-1)
        pca = PCA(n_components=min(n_components, len(V_cols)), random_state=42)
        comp = pca.fit_transform(df_V)
        for i in range(comp.shape[1]):
            df[f'V_pca_{i}'] = comp[:, i]
    return df

def select_features(df, max_features=100):
    # heuristic: numeric and engineered features
    cols = df.select_dtypes(include=['number']).columns.tolist()
    # drop id and target
    cols = [c for c in cols if c not in ['TransactionID','isFraud']]
    return cols[:max_features]
