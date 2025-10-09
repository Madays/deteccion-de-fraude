# Fraud Detection Project Documentation

## Overview

This project implements an advanced fraud detection system using machine learning techniques on the IEEE-CIS Fraud Detection dataset. The system combines supervised learning with unsupervised anomaly detection to identify fraudulent transactions. It is designed as a template for experimentation and production use, with a modular architecture for easy maintenance and extension.

## Architecture

### Directory Structure

```
deteccion-de-fraude/
├── data/                      # Dataset files
│   ├── train_transaction.csv  # Transaction data for training
│   ├── train_identity.csv     # Identity data for training
│   ├── test_transaction.csv   # Test transaction data
│   ├── test_identity.csv      # Test identity data
│   └── new_transactions.csv   # New transactions for prediction
├── models/                    # Trained models and encoders
│   ├── lightgbm_fold1.pkl     # LightGBM model fold 1
│   ├── lightgbm_fold2.pkl     # LightGBM model fold 2
│   ├── lightgbm_fold3.pkl     # LightGBM model fold 3
│   ├── lightgbm_fold4.pkl     # LightGBM model fold 4
│   ├── lightgbm_fold5.pkl     # LightGBM model fold 5
│   ├── isoforest.joblib       # Isolation Forest model
│   ├── target_encoder.joblib  # Target encoder for categorical features
│   └── selected_features.pkl  # List of selected features
├── outputs/                   # Results and metrics
│   ├── evaluation_metrics.json # Training metrics
│   └── new_predictions.csv    # Predictions on new data
├── src/                       # Source code modules
│   ├── preprocess.py          # Data preprocessing functions
│   ├── feature_engineering.py # Feature engineering functions
│   ├── train_lightgbm.py      # LightGBM training functions
│   ├── anomaly_isoforest.py   # Isolation Forest functions
│   └── utils.py               # Utility functions
├── main_train.py              # Main training script
├── predict_new.py             # Prediction script
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

### System Architecture

The system follows a pipeline architecture with the following stages:

1. **Data Ingestion**: Load and merge transaction and identity data
2. **Preprocessing**: Clean data, handle missing values, encode categorical features
3. **Feature Engineering**: Create new features through transformations, aggregations, and dimensionality reduction
4. **Model Training**: Train ensemble of LightGBM models and Isolation Forest
5. **Prediction**: Apply trained models to new data and combine predictions
6. **Evaluation**: Assess model performance using AUC and other metrics

### Key Components

#### Main Scripts

- **`main_train.py`**: Orchestrates the entire training pipeline from data loading to model saving
- **`predict_new.py`**: Handles prediction on new transactions using saved models

#### Modules

- **`src/preprocess.py`**: Contains functions for data loading, merging, cleaning, imputation, and encoding
- **`src/feature_engineering.py`**: Implements feature transformations including log scaling, time features, aggregations, and PCA
- **`src/train_lightgbm.py`**: Provides k-fold cross-validation training for LightGBM with early stopping
- **`src/anomaly_isoforest.py`**: Trains and applies Isolation Forest for unsupervised anomaly detection
- **`src/utils.py`**: Utility functions for directory creation, JSON saving, and model serialization

## Code Documentation

### main_train.py

This script executes the complete training pipeline.

**Key Functions:**

- `main(data_dir, out_dir, models_dir)`: Main training function that:
  - Loads and merges data
  - Applies preprocessing and feature engineering
  - Trains LightGBM with 5-fold cross-validation
  - Trains Isolation Forest
  - Saves models and evaluation metrics

**Workflow:**

1. Load training data from `data_dir`
2. Merge transaction and identity data
3. Apply basic cleaning and imputation
4. Perform feature engineering (log transform, time features, aggregations, PCA)
5. Target encode categorical features
6. Create train/validation split
7. Select top features
8. Train LightGBM ensemble
9. Train Isolation Forest
10. Evaluate and save results

### predict_new.py

This script applies trained models to new transactions.

**Workflow:**

1. Load new transaction data
2. Apply preprocessing (cleaning, imputation, encoding)
3. Perform feature engineering
4. Load trained models
5. Generate predictions from LightGBM and Isolation Forest
6. Combine predictions with custom logic
7. Save results to CSV

### src/preprocess.py

Contains data preprocessing utilities.

**Functions:**

- `load_data(data_dir)`: Loads transaction and identity CSV files
- `merge_identity(tx, idf)`: Merges transaction and identity data on TransactionID
- `basic_clean(df)`: Normalizes column names, drops columns with single unique value
- `basic_impute(df, num_fill=-1, cat_fill='missing')`: Fills missing values in numeric and categorical columns
- `target_encode_train(df, categorical_cols, target_col='isFraud')`: Fits and applies target encoding
- `apply_target_encoding(df, enc, categorical_cols)`: Applies pre-fitted target encoding
- `create_holdout(df, test_size=0.2, random_state=42)`: Creates stratified train/validation split

### src/feature_engineering.py

Implements feature engineering techniques.

**Functions:**

- `log_transform_amount(df, col='TransactionAmt')`: Applies log(1+x) transformation to transaction amounts
- `create_time_features(df, dt_col='TransactionDT')`: Extracts hour and day features from timestamp
- `card_agg_features(df)`: Creates card-based aggregations (mean, std of amounts per card)
- `reduce_V_features_pca(df, n_components=20)`: Applies PCA to V-prefixed features for dimensionality reduction
- `select_features(df, max_features=100)`: Selects numeric features, excluding IDs and targets

### src/train_lightgbm.py

Handles LightGBM model training with cross-validation.

**Functions:**

- `train_lgbm_kfold(X, y, features, n_splits=5, params=None, seed=42)`: Trains 5-fold LightGBM ensemble with early stopping
- `evaluate_binary(y_true, y_pred_prob, threshold=0.5)`: Computes precision, recall, and F1-score

**Training Details:**

- Uses StratifiedKFold for balanced class distribution
- Implements early stopping with 50 rounds patience
- Returns trained models, out-of-fold predictions, and overall AUC

### src/anomaly_isoforest.py

Implements unsupervised anomaly detection.

**Functions:**

- `train_isolation_forest(X, contamination=0.01, random_state=42)`: Trains Isolation Forest model
- `score_with_isoforest(iso, X)`: Computes anomaly scores for new data

**Notes:**

- Contamination parameter set to 0.01 (1% expected anomalies)
- Decision function returns scores where lower values indicate higher anomaly likelihood

### src/utils.py

Provides utility functions for file operations.

**Functions:**

- `ensure_dir(path)`: Creates directory if it doesn't exist
- `save_json(obj, path)`: Saves dictionary as JSON file
- `save_model(obj, path)`: Saves model using joblib

## Design Decisions and Rationale

### Why This Architecture?

1. **Modularity**: Separating concerns into distinct modules (preprocessing, feature engineering, training) allows for easy testing, debugging, and extension. Each module can be developed and maintained independently.

2. **Pipeline Design**: The linear pipeline approach ensures reproducibility and makes it easy to add or modify steps without affecting the entire system.

3. **Combination of Supervised and Unsupervised Learning**:

   - LightGBM provides supervised learning on labeled fraud data
   - Isolation Forest detects novel anomalies not captured by labeled data
   - Combining both approaches increases robustness against unknown fraud patterns

4. **Feature Engineering Focus**:

   - The IEEE dataset has many features, including high-dimensional V features
   - Log transformation handles skewed transaction amounts
   - Time features capture temporal patterns
   - Card aggregations detect unusual behavior relative to card history
   - PCA reduces dimensionality while preserving variance

5. **Cross-Validation**: K-fold training prevents overfitting and provides more reliable performance estimates compared to single train/test split.

6. **Target Encoding**: Handles high-cardinality categorical features by encoding them based on fraud rates, which is more effective than one-hot encoding for this use case.

7. **Model Serialization**: Using joblib for model saving ensures compatibility and efficient loading for predictions.

### Performance Considerations

- LightGBM chosen for its speed and performance on tabular data
- Isolation Forest selected for its ability to handle high-dimensional data without assumptions about data distribution
- Parallel processing enabled where possible (n_jobs=-1)

### Scalability

- The pipeline can handle large datasets by processing in chunks if needed
- Feature selection limits the number of features to prevent overfitting and reduce computation time
- Models are saved separately to allow loading only what's needed for prediction

### Limitations and Future Improvements

- Current implementation assumes availability of identity data; graceful handling when missing
- Hyperparameter tuning could be added for better performance
- More advanced feature engineering (e.g., interaction features, embedding layers)
- Ensemble methods combining multiple algorithms
- Real-time prediction capabilities
- Model monitoring and retraining pipelines

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning utilities and Isolation Forest
- lightgbm: Gradient boosting framework
- category_encoders: Target encoding
- joblib: Model serialization
- tqdm: Progress bars (not used in current code but listed)
- shap: Model interpretability (not used but potentially useful)
- matplotlib/seaborn: Visualization (for analysis)

## Usage

1. Place dataset files in `data/` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python main_train.py`
4. For predictions: `python predict_new.py`

See README.md for detailed instructions.
