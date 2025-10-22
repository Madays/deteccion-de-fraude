# 🧠 Documentación de Calibración de Parámetros — Detección de Fraude

Excelente pregunta 👏  
Este documento explica **en español y con detalle** cómo calibrar los **parámetros e hiperparámetros** del modelo de detección de fraude, además de la configuración de validación para obtener un rendimiento más realista y estable.

---

## 🧩 1. Estrategia de división de datos (Data Splitting & Validation)

### 🔹 División temporal ("Time-based Split")

- **Qué hace:** divide los datos según el **tiempo** (por ejemplo, por el día de la transacción).
- **Cómo se define:** se toma el percentil 80 del campo `day` → todo lo anterior al día 58 se usa para **entrenar**, y lo posterior para **validar**.
- **Por qué se hace:** esto **evita fugas de datos del futuro** ("data leakage"), asegurando que el modelo solo vea transacciones anteriores, no futuras.

**Ejemplo:**

- Transacciones del día 1 al 58 → entrenamiento (80%)
- Transacciones del día 58 al 73 → validación (20%)

📘 Esto imita cómo el modelo se usará en producción: entrenas con el pasado y predices el futuro.

---

### 🔹 Validación cruzada estratificada (Stratified K-Fold)

- **Qué es:** divide los datos de entrenamiento en 5 partes (folds), manteniendo la proporción de fraudes y no fraudes en cada una.
- **Propósito:** obtener un rendimiento más estable y evitar sobreajuste ("overfitting").
- **Parámetros:**
  - `n_splits = 5`
  - `shuffle = True` (mezcla las filas)
  - `random_state = 42` (reproducibilidad)

📘 Esto permite obtener un valor medio del rendimiento del modelo en distintos subconjuntos de los datos (OOF = Out Of Fold).

---

## ⚙️ 2. Parámetros de _Feature Engineering_

### 🔹 PCA (Reducción de dimensionalidad)

- **Qué hace:** toma las columnas `V1, V2, ..., V339` y las resume en **20 componentes** principales.
- **Objetivo:** reducir ruido y correlaciones redundantes, acelerando el entrenamiento.
- **Modo entrenamiento:** el PCA se ajusta solo con los datos de _train_ (para evitar fuga de datos).
- **Valores faltantes:** se rellenan con `-1`.

📘 Ejemplo: en lugar de 339 columnas "V", se usan 20 que resumen la variación principal.

---

### 🔹 Selección de features (Feature Selection)

- **Método:** se eligen las **300 columnas con mayor varianza** (las más informativas).
- **Exclusión:** se ignoran `TransactionID`, `isFraud` y columnas no numéricas.
- **Orden:** de mayor a menor varianza.

📘 Esto elimina columnas con poca información o ruido, mejorando velocidad y generalización.

---

### 🔹 Codificación categórica (Target Encoding)

- **Qué hace:** reemplaza categorías (por ejemplo, "tipo de dispositivo") por la **probabilidad media de fraude** asociada a esa categoría.
- **Parámetros:**
  - `smoothing = 0.3` (suaviza el efecto de categorías con pocos datos)
  - Solo se codifican columnas con menos de 200 valores únicos.
  - Categorías desconocidas → se reemplazan por `'missing'`.

📘 Ejemplo:  
Si "browser = Chrome" tiene 2% de fraude y "browser = IE" 6%, se reemplaza el texto por esos valores numéricos.

---

## ⚡ 3. Hiperparámetros de LightGBM

```python
lgb_params = {
    'objective': 'binary',        # clasificación binaria (fraude o no)
    'boosting_type': 'gbdt',      # tipo de boosting
    'metric': 'auc',              # métrica principal
    'learning_rate': 0.03,        # velocidad de aprendizaje
    'num_leaves': 128,            # complejidad de los árboles
    'feature_fraction': 0.8,      # fracción de features usadas por iteración
    'bagging_fraction': 0.8,      # fracción de datos usados por iteración
    'bagging_freq': 5,            # frecuencia del bagging
    'lambda_l1': 0.5,             # regularización L1
    'lambda_l2': 0.5,             # regularización L2
    'min_data_in_leaf': 30,       # mínimo de datos por hoja
    'verbosity': -1,
    'seed': 42,
    'n_jobs': -1
}
```

### Entrenamiento:

num_boost_round = 4000 (máximo número de árboles)

early_stopping_rounds = 100 (si no mejora el AUC por 100 iteraciones, se detiene)

Evalúa cada 200 rondas para monitorear progreso

📘 Este balancea velocidad y generalización del modelo.

---

## 🧩 4. Isolation Forest (para detectar anomalías)

Detecta comportamientos raros que podrían ser fraudes.

Parámetros:

n_estimators = 200

contamination = 0.01 → espera que el 1% sean anómalos

random_state = 42

n_jobs = -1 (usa todos los núcleos)

📘 Este paso ayuda a crear una feature extra que mide "qué tan anómala" es una transacción.

---

## 🎯 5. Calibración del Umbral (Threshold)

El modelo predice probabilidades (ej. 0.82 = posible fraude), pero hay que definir a partir de qué valor se considera "fraude".

Se busca el umbral óptimo que maximiza el F1-score
(balance entre precisión y recall).

Fórmula:

F1 = 2 _ (precision _ recall) / (precision + recall)

En tu modelo actual:

Threshold óptimo = 0.2153

F1 = 0.7531

📘 Es decir: si la probabilidad > 0.2153, el modelo marca la transacción como fraude.

---

## 📊 6. Métricas Actuales

| Métrica               | Descripción                                                  | Valor  |
| --------------------- | ------------------------------------------------------------ | ------ |
| OOF AUC (Out-of-Fold) | AUC promedio en los folds de entrenamiento                   | 0.9530 |
| Holdout AUC           | AUC en el conjunto final de validación (simula datos nuevos) | 0.9211 |
| Best F1               | Mejor balance entre precisión y recall                       | 0.7531 |

📘 Interpretación:

Tu modelo generaliza bien (AUC 0.92 en datos no vistos).

Pero todavía hay espacio para mejorar el F1, afinando el umbral o los pesos de clases.

---

## 🧠 7. Recomendaciones para Calibración

### 🔹 Ajustar hiperparámetros

Prueba combinaciones para encontrar el mejor equilibrio entre complejidad y sobreajuste:

learning_rate: [0.01, 0.03, 0.05, 0.1]

num_leaves: [64, 128, 256]

feature_fraction / bagging_fraction: [0.6, 0.8, 1.0]

lambda_l1 / lambda_l2: [0.1, 0.5, 1.0]

### 🔹 Tuning de features

PCA: prueba 10, 20, 30, 50 componentes

Selección de features: 200, 300 o 500 columnas

Target encoding smoothing: 0.1, 0.3, 0.5, 1.0

### 🔹 Validación realista

Mantén la división temporal

Considera validación con ventana deslizante (rolling window) para detectar drift (cambios de comportamiento en el tiempo)

Evalúa no solo F1, sino también costo-beneficio (ej. pérdida por falsos negativos)

---

## 🏁 En resumen

Este documento describe cómo controlar y optimizar todos los aspectos del entrenamiento:

- División temporal → evita fugas de datos
- Validación estratificada → robustez
- PCA + selección → eficiencia
- Target encoding → mejor manejo de categorías
- LightGBM optimizado → buen AUC
- Umbral calibrado → buen F1
- Ajustes finos → mejor rendimiento realista
