# Detección de Fraude con Machine Learning

Este proyecto implementa un sistema avanzado de detección de fraude en transacciones financieras utilizando técnicas de machine learning. Combina aprendizaje supervisado con detección de anomalías no supervisada para identificar transacciones fraudulentas en el dataset IEEE-CIS Fraud Detection.

## 📋 Descripción del Proyecto

El sistema está diseñado como una plantilla para experimentación y uso en producción, con una arquitectura modular que facilita el mantenimiento y la extensión. Utiliza el dataset de Kaggle IEEE-CIS Fraud Detection, que contiene datos de transacciones e-commerce con características anónimas y etiquetas de fraude.

### 🎯 Objetivo

Desarrollar un modelo robusto que pueda detectar transacciones fraudulentas en tiempo real, combinando múltiples técnicas de machine learning para maximizar la precisión y minimizar los falsos positivos.

## 📊 Dataset Utilizado

**IEEE-CIS Fraud Detection Dataset** ([Enlace a Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data?select=train_transaction.csv))

### Estructura de Datos:

- **`train_transaction.csv`**: Datos principales de transacciones (~590k filas)

  - `TransactionID`: ID único de la transacción
  - `isFraud`: Etiqueta objetivo (0 = legítima, 1 = fraudulenta)
  - `TransactionDT`: Timestamp de la transacción (segundos desde referencia)
  - `TransactionAmt`: Monto de la transacción
  - `ProductCD`: Código del producto
  - `card1-card6`: Información de la tarjeta (anónima)
  - `addr1-addr2`: Direcciones del comprador
  - `dist1-dist2`: Distancias
  - `P_emaildomain`, `R_emaildomain`: Dominios de email
  - `C1-C14`: Features de conteo
  - `D1-D15`: Features de tiempo/delay
  - `M1-M9`: Features de matching
  - `V1-V339`: Features anónimas (principalmente numéricas)

- **`train_identity.csv`**: Datos de identidad (~140k filas)
  - Información del dispositivo y navegador
  - `id_01` to `id_38`: Features de identidad anónimas
  - `DeviceType`, `DeviceInfo`: Tipo e info del dispositivo

### Distribución de Clases:

- Transacciones legítimas: ~96.5%
- Transacciones fraudulentas: ~3.5%
- Ratio de desbalance: ~27:1

## 🏗️ Arquitectura del Sistema

### Estructura de Directorios

```
deteccion-de-fraude/
├── data/                      # Archivos de datos
│   ├── train_transaction.csv  # Datos de entrenamiento
│   ├── train_identity.csv     # Datos de identidad entrenamiento
│   ├── test_transaction.csv   # Datos de prueba
│   ├── test_identity.csv      # Datos de identidad prueba
│   └── new_transactions.csv   # Nuevas transacciones para predicción
├── models/                    # Modelos entrenados y encoders
│   ├── lightgbm_fold[1-5].pkl # Modelos LightGBM (5-fold CV)
│   ├── isoforest.joblib       # Modelo Isolation Forest
│   ├── target_encoder_full.joblib # Encoder categórico global
│   ├── pca_full.joblib        # PCA para features V
│   ├── pca_fold[1-5].joblib   # PCA por fold
│   ├── selected_features.pkl  # Lista de features seleccionadas
│   └── decision_threshold.pkl # Umbral óptimo de decisión
├── outputs/                   # Resultados y métricas
│   ├── evaluation_metrics.json # Métricas de evaluación
│   └── new_predictions.csv    # Predicciones en nuevas transacciones
├── src/                       # Código fuente modular
│   ├── preprocess.py          # Funciones de preprocesamiento
│   ├── feature_engineering.py # Ingeniería de features
│   └── anomaly_isoforest.py   # Detección de anomalías
├── main_train.py              # Script principal de entrenamiento
├── predict_new.py             # Script de predicción
├── requirements.txt           # Dependencias Python
├── docs.md                    # Documentación técnica detallada
└── README.md                  # Este archivo
```

### Pipeline de Procesamiento

1. **Ingesta de Datos**: Carga y merge de datos de transacciones e identidad
2. **Preprocesamiento**: Limpieza, imputación de valores faltantes, encoding categórico
3. **Ingeniería de Features**: Transformaciones, agregaciones, reducción dimensional
4. **Entrenamiento**: Modelos supervisados (LightGBM) + no supervisados (Isolation Forest)
5. **Predicción**: Aplicación de modelos entrenados a nuevos datos
6. **Evaluación**: Métricas de performance y validación

## ⚙️ Requisitos para Ejecutar el Entrenamiento

### Dependencias

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
category_encoders>=2.3.0
joblib>=1.1.0
tqdm>=4.62.0
shap>=0.40.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Instalación

```bash
pip install -r requirements.txt
```

### Requisitos de Hardware

- **RAM**: Mínimo 16GB, recomendado 32GB+
- **CPU**: Multi-core recomendado (el código usa `n_jobs=-1`)
- **Almacenamiento**: ~5GB para datos + modelos
- **GPU**: Opcional (LightGBM puede usar GPU para entrenamiento más rápido)

### Requisitos de Datos

1. Descargar el dataset de Kaggle y colocar los archivos en `data/`:

   - `train_transaction.csv`
   - `train_identity.csv`
   - `test_transaction.csv` (opcional)
   - `test_identity.csv` (opcional)

2. Para predicciones, crear `data/new_transactions.csv` con el mismo formato que `train_transaction.csv` (sin columna `isFraud`)

## 🚀 Cómo Usar el Sistema

### 1. Preparación de Datos

```bash
# Descargar dataset de Kaggle y colocar en data/
# Asegurarse de que los archivos estén en el directorio correcto
ls data/
# train_transaction.csv  train_identity.csv  test_transaction.csv  test_identity.csv
```

### 2. Entrenamiento del Modelo

```bash
python main_train.py
```

**Qué hace el entrenamiento:**

- Carga y preprocesa los datos
- Aplica ingeniería de features
- Entrena 5 modelos LightGBM con cross-validation estratificada
- Entrena Isolation Forest para detección de anomalías
- Calcula umbral óptimo de decisión
- Guarda todos los modelos y métricas

**Tiempo estimado**: ~30-60 minutos dependiendo del hardware

### 3. Predicción en Nuevas Transacciones

```bash
python predict_new.py
```

**Proceso de predicción:**

- Carga nuevas transacciones desde `data/new_transactions.csv`
- Aplica el mismo preprocesamiento y feature engineering
- Genera predicciones con LightGBM + Isolation Forest
- Combina scores con lógica de decisión
- Guarda resultados en `outputs/new_predictions.csv`

## 🔧 Detalles del Entrenamiento

### Estrategia de Validación

- **Time-based Split**: Entrenamiento en 80% temporal temprano, validación en 20% temporal tardío
- **5-Fold Cross-Validation**: Dentro del conjunto de entrenamiento para evaluación robusta
- **Prevención de Data Leakage**: Encoding y PCA ajustados por fold para evitar contaminación

### Features Engineering

1. **Transformaciones Básicas**:

   - Log-transform del monto de transacción (`TransactionAmt_log`)
   - Features temporales: `hour`, `day` desde `TransactionDT`

2. **Agregaciones por Tarjeta**:

   - Media y desviación estándar de montos por `card1`
   - Ratio del monto actual vs. media histórica

3. **Reducción Dimensional**:

   - PCA en features V (339 → 20 componentes)
   - Selección de top 300 features por varianza

4. **Encoding Categórico**:
   - Target Encoding para categorías con <200 valores únicos
   - Smoothing para evitar overfitting

### Modelos Utilizados

#### LightGBM (Supervisado)

- **Objetivo**: `binary` (clasificación binaria)
- **Métrica**: AUC
- **Hiperparámetros**:
  - `learning_rate`: 0.03
  - `num_leaves`: 128
  - `feature_fraction`: 0.8
  - `bagging_fraction`: 0.8
  - `lambda_l1/lambda_l2`: 0.5
  - `min_data_in_leaf`: 30
- **Entrenamiento**: 4000 iteraciones con early stopping (100 rondas)

#### Isolation Forest (No Supervisado)

- **Contaminación**: 1% (esperado ratio de anomalías)
- **Estimators**: 200 árboles
- **Random State**: 42

### Métricas de Performance

- **OOF AUC**: 0.953 (Cross-validation en entrenamiento)
- **Holdout AUC**: 0.921 (Validación temporal)
- **Best Threshold**: 0.215 (óptimo para F1-score)
- **Best F1-Score**: 0.753

### Criterios de Entrenamiento

1. **Robustez Temporal**: Validación en datos futuros para simular producción
2. **Prevención de Overfitting**: Regularización L1/L2, feature selection, early stopping
3. **Manejo de Desbalance**: Estratificación en CV, evaluación con F1-score
4. **Eficiencia Computacional**: PCA para reducción dimensional, paralelización
5. **Interpretabilidad**: Features seleccionadas por varianza, posibilidad de SHAP

## 📈 Resultados y Evaluación

### Métricas Principales

```json
{
  "oof_auc": 0.9529786660279241,
  "val_auc": 0.9210964511516676,
  "best_threshold": 0.2152629992636282,
  "best_f1": 0.753071671862922
}
```

### Interpretación

- **AUC Alto**: Excelente capacidad discriminativa
- **F1-Score**: Buen balance precision/recall para clase minoritaria
- **Threshold Óptimo**: Calibrado para maximizar F1 en datos de validación

### Predicciones en Nuevas Transacciones

El sistema genera:

- `fraud_probability`: Score de LightGBM (0-1)
- `anomaly_score`: Score de Isolation Forest (más negativo = más anómalo)
- `is_fraud_pred`: Decisión final (1 = fraudulento, 0 = legítimo)

**Lógica de Decisión Final**:

```python
fraud_flags = (fraud_probability >= threshold) | (anomaly_score < -0.2)
```

## 🔍 Análisis del Código

### Scripts Principales

#### `main_train.py`

Orquesta todo el pipeline de entrenamiento:

- Carga y merge de datos
- Preprocesamiento básico
- Split temporal train/holdout
- Feature engineering
- Entrenamiento con CV
- Evaluación y guardado

#### `predict_new.py`

Pipeline de inferencia:

- Carga nuevas transacciones
- Aplicación de transformaciones aprendidas
- Predicciones ensemble
- Guardado de resultados

### Módulos Core

#### `src/preprocess.py`

- `basic_clean()`: Limpieza de datos, normalización
- `basic_impute()`: Imputación de valores faltantes
- `fit_target_encoder()`: Ajuste de encoding categórico
- `apply_target_encoding()`: Aplicación de encoding

#### `src/feature_engineering.py`

- `log_transform_amount()`: Transformación logarítmica
- `create_time_features()`: Extracción de features temporales
- `card_agg_features()`: Agregaciones por tarjeta
- `reduce_V_features_pca()`: Reducción dimensional con PCA
- `select_features()`: Selección por varianza

#### `src/anomaly_isoforest.py`

- `train_isolation_forest()`: Entrenamiento del modelo no supervisado.

## 🎯 Mejoras Futuras

1. **Hiperparameter Tuning**: Optimización automática de parámetros
2. **Features Adicionales**: Interacciones, embeddings, features temporales avanzadas
3. **Modelos Ensemble**: Combinación de múltiples algoritmos
4. **Real-time Scoring**: API para predicciones en tiempo real
5. **Monitoring**: Sistema de monitoreo de performance en producción
6. **Explainability**: Análisis SHAP para interpretabilidad

## 📝 Notas Importantes

- El sistema está optimizado para el dataset IEEE-CIS específico
- Los features V son anónimos pero cruciales para la performance
- El threshold óptimo puede requerir ajuste según el caso de uso específico
- Para producción, considerar validación adicional y monitoring continuo

## 🤝 Contribución

Este proyecto está diseñado para ser extensible. Las mejoras sugeridas incluyen:

- Nuevos algoritmos de ML
- Features engineering adicionales
- Mejores estrategias de validación
- Optimizaciones de performance

## 📄 Licencia

Este proyecto es para fines educativos y de investigación. El dataset IEEE-CIS tiene sus propias condiciones de uso a través de Kaggle.
