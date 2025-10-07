import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, CountEncoder
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    tx = pd.read_csv(f"{data_dir}/train_transaction.csv", low_memory=False)
    try:
        idf = pd.read_csv(f"{data_dir}/train_identity.csv", low_memory=False)
    except FileNotFoundError:
        idf = None
    return tx, idf

def merge_identity(tx, idf):
    if idf is None:
        return tx
    return tx.merge(idf, how='left', on='TransactionID')

def basic_clean(df):
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    # Example: dropcols with single unique value
    nunique = df.nunique(dropna=False)
    drop = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=drop)
    return df

def basic_impute(df, num_fill=-1, cat_fill='missing'):
    # numeric columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[num_cols] = df[num_cols].fillna(num_fill)
    df[cat_cols] = df[cat_cols].fillna(cat_fill)
    return df

def target_encode_train(df, categorical_cols, target_col='isFraud'):
    enc = TargetEncoder(cols=categorical_cols, smoothing=0.3)
    df[categorical_cols] = enc.fit_transform(df[categorical_cols], df[target_col])
    return df, enc

def apply_target_encoding(df, enc, categorical_cols):
    df[categorical_cols] = enc.transform(df[categorical_cols])
    return df

def create_holdout(df, test_size=0.2, random_state=42):
    train, val = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['isFraud'])
    return train.reset_index(drop=True), val.reset_index(drop=True)
