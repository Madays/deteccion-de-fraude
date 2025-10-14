import pandas as pd
import numpy as np
import joblib
from category_encoders import TargetEncoder

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Quitar columnas completamente vacías
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        df = df.drop(columns=null_cols)
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # Normalizar strings a lower-case
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.lower()
    return df

def basic_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # num -> median, cat -> 'missing'
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=[np.number]).columns:
        df[col] = df[col].fillna('missing')
    return df

def fit_target_encoder(df: pd.DataFrame, cat_cols: list, target_col: str):
    """
    Ajusta TargetEncoder en cat_cols usando sólo filas con target no nulo.
    Retorna (encoder, df_transformed).
    """
    df = df.copy()
    # Filtrar filas con target no nulo
    mask = df[target_col].notna()
    df_fit = df.loc[mask, cat_cols].copy()
    y_fit = df.loc[mask, target_col].values

    # Si no hay columnas categóricas, devolvemos encoder None
    if len(cat_cols) == 0:
        encoder = None
        return encoder, df

    enc = TargetEncoder(cols=cat_cols, smoothing=0.3)
    enc.fit(df_fit, y_fit)
    # añadir metadata
    enc.cols = cat_cols
    # transformar todo el df (incluyendo filas que no se usaron para fit)
    df[cat_cols] = enc.transform(df[cat_cols])
    return enc, df

def apply_target_encoding(df: pd.DataFrame, encoder, cat_cols: list):
    """
    Aplica encoder a df. Si falta alguna columna del encoder, la crea con 'missing'.
    """
    df = df.copy()
    if encoder is None or len(cat_cols) == 0:
        return df
    missing = [c for c in cat_cols if c not in df.columns]
    if missing:
        fill = {c: 'missing' for c in missing}
        fill_df = pd.DataFrame(fill, index=df.index)
        df = pd.concat([df, fill_df], axis=1)
    # ahora transformar (encoder.transform espera DataFrame con las columnas)
    df[cat_cols] = encoder.transform(df[cat_cols])
    return df
