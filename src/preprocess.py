# Este archivo contiene funciones para el preprocesamiento de datos en el proyecto de detección de fraude.
# El preprocesamiento prepara los datos crudos para que puedan ser usados por los modelos de machine learning.
# Importamos bibliotecas necesarias: pandas para manejo de datos, numpy para operaciones numéricas,
# category_encoders para codificar variables categóricas, y sklearn para dividir datos.
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, CountEncoder
from sklearn.model_selection import train_test_split

# Función para cargar los datos desde archivos CSV.
# Recibe el directorio donde están los datos y devuelve las transacciones y datos de identidad.
def load_data(data_dir):
    # Cargamos el archivo de transacciones de entrenamiento. 'low_memory=False' evita problemas con tipos de datos mixtos.
    tx = pd.read_csv(f"{data_dir}/train_transaction.csv", low_memory=False)
    # Intentamos cargar el archivo de identidad de entrenamiento.
    try:
        idf = pd.read_csv(f"{data_dir}/train_identity.csv", low_memory=False)
    # Si el archivo no existe, asignamos None (algunos datasets pueden no tener identidad).
    except FileNotFoundError:
        idf = None
    # Devolvemos las transacciones y la identidad (o None si no hay).
    return tx, idf

# Función para fusionar los datos de transacciones con los de identidad.
# Si no hay datos de identidad, devuelve solo las transacciones.
def merge_identity(tx, idf):
    # Si idf es None, no hay datos de identidad, así que devolvemos tx tal cual.
    if idf is None:
        return tx
    # Fusionamos tx e idf usando 'left join' en la columna 'TransactionID'.
    # Esto significa que mantenemos todas las filas de tx y agregamos info de idf donde coincida.
    return tx.merge(idf, how='left', on='TransactionID')

# Función para limpieza básica de los datos.
# Elimina columnas innecesarias y normaliza nombres.
def basic_clean(df):
    # Normalizamos los nombres de las columnas eliminando espacios en blanco al inicio y fin.
    df.columns = [c.strip() for c in df.columns]
    # Calculamos el número de valores únicos en cada columna, incluyendo NaN.
    nunique = df.nunique(dropna=False)
    # Identificamos columnas que tienen solo 1 valor único (son constantes, no aportan info).
    drop = nunique[nunique <= 1].index.tolist()
    # Eliminamos esas columnas del dataframe.
    df = df.drop(columns=drop)
    # Devolvemos el dataframe limpio.
    return df

# Función para imputar (rellenar) valores faltantes de manera básica.
# Rellena numéricos con un valor por defecto y categóricos con 'missing'.
def basic_impute(df, num_fill=-1, cat_fill='missing'):
    # Seleccionamos las columnas numéricas (de tipo número).
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Seleccionamos las columnas categóricas (de tipo objeto, como strings).
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Rellenamos los valores faltantes en columnas numéricas con num_fill (-1 por defecto).
    df[num_cols] = df[num_cols].fillna(num_fill)
    # Rellenamos los valores faltantes en columnas categóricas con cat_fill ('missing' por defecto).
    df[cat_cols] = df[cat_cols].fillna(cat_fill)
    # Devolvemos el dataframe con valores imputados.
    return df

# Función para aplicar target encoding en el conjunto de entrenamiento.
# Target encoding convierte categorías en números basados en la relación con la variable objetivo (isFraud).
def target_encode_train(df, categorical_cols, target_col='isFraud'):
    # Creamos un codificador TargetEncoder con suavizado (smoothing) para evitar sobreajuste.
    enc = TargetEncoder(cols=categorical_cols, smoothing=0.3)
    # Ajustamos el codificador con los datos de entrenamiento y transformamos las columnas categóricas.
    df[categorical_cols] = enc.fit_transform(df[categorical_cols], df[target_col])
    # Devolvemos el dataframe transformado y el codificador (para usar en test después).
    return df, enc

# Función para aplicar el target encoding ya entrenado en nuevos datos (como test).
# Usa el codificador guardado del entrenamiento.
def apply_target_encoding(df, enc, categorical_cols):
    # Transformamos las columnas categóricas usando el codificador ya ajustado.
    df[categorical_cols] = enc.transform(df[categorical_cols])
    # Devolvemos el dataframe transformado.
    return df

# Función para crear una división de entrenamiento y validación (holdout).
# Divide los datos manteniendo la proporción de la clase objetivo.
def create_holdout(df, test_size=0.2, random_state=42):
    # Usamos train_test_split para dividir en train y val.
    # test_size=0.2 significa 20% para validación, 80% para entrenamiento.
    # stratify=df['isFraud'] asegura que la proporción de fraudes sea igual en ambos sets.
    # random_state=42 para reproducibilidad (mismos resultados cada vez).
    train, val = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['isFraud'])
    # Reseteamos los índices y devolvemos los dataframes.
    return train.reset_index(drop=True), val.reset_index(drop=True)
