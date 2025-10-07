# Fraude - Advanced Training Template (LightGBM + IsolationForest)

Plantilla avanzada para entrenar modelos sobre el dataset IEEE-CIS Fraud Detection (train_transaction.csv + train_identity.csv).
Estructura de la plantilla, scripts para preprocesamiento, feature engineering, entrenamiento supervisado (LightGBM) y no supervisado (IsolationForest).

## Requisitos
- Colocar `train_transaction.csv` y `train_identity.csv` en la carpeta `data/`.
- Python 3.9+
- Instalar dependencias:
```
pip install -r requirements.txt
```

## Estructura
```
fraude_advanced_template/
├── data/                      # coloque aquí train_transaction.csv, train_identity.csv
├── models/                    # modelos guardados
├── notebooks/                 
├── outputs/                   # métricas y artefactos
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_lightgbm.py
│   ├── anomaly_isoforest.py
│   └── utils.py
├── main_train.py
├── requirements.txt
└── README.md
```

## Cómo usar
1. Poner archivos CSV en `data/`.
2. Ejecutar:
```
python main_train.py --data_dir data --out_dir outputs --models_dir models
```
El pipeline realizará preprocesamiento, feature engineering, entrenará LightGBM con K-Fold y entrenará IsolationForest. Los modelos y métricas se guardarán en `models/` y `outputs/`.

Nota: Esta plantilla es para experimentación y como punto de partida. Ajustes y validación deben realizarse antes de usar en producción.
