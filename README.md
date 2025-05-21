# Churn Predictor

A machine learning project for predicting customer churn in banking services using XGBoost and SMOTE. This project implements a comprehensive pipeline for data processing, feature engineering, and model training to predict customer churn with high accuracy.

## Project Structure

```
.
├── data/               # Data directory
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks
├── reports/           # Analysis reports and visualizations
├── src/               # Source code
│   ├── config.py              # Configuration settings and paths
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── feature_engineering.py # Feature creation and transformation
│   ├── model.py              # Model training and evaluation
│   ├── train.py             # Training pipeline
│   └── utils.py             # Utility functions
└── requirements.txt   # Project dependencies
```

## Features

### Data Processing
- Proper data type conversion and validation
- Train-test split with stratification

### Feature Engineering
- Age grouping (18-29, 30-44, 45-59, 60+)
- Tenure grouping (0-2, 3-5, 6-10, 10+ years)
- Balance to salary ratio
- Zero balance indicator
- High balance but low salary flag
- Geography-Gender interaction features
- Credit utilization proxy

### Model Pipeline
- SMOTE for handling class imbalance
- XGBoost classifier with optimized hyperparameters
- Comprehensive model evaluation metrics
- Model persistence and loading capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GunDalf101/Churn_Predictor.git
cd churn-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your raw data file (`Churn_Modelling.csv`) in the `data/raw/` directory.

### Training the Model

```python
from src.model import ChurnPredictor
from src.data_processing import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Initialize and train the model
predictor = ChurnPredictor()
predictor.train(X_train, y_train)

# Evaluate the model
metrics = predictor.evaluate(X_test, y_test)
print(metrics)

# Save the model
predictor.save_model()
```

### Making Predictions

```python
# Load a saved model
predictor = ChurnPredictor.load_model()

# Make predictions
predictions = predictor.predict(new_data)
probabilities = predictor.predict_proba(new_data)
```

## Model Details

### Feature Sets
- Categorical Features:
  - Geography
  - Gender
  - HasCrCard
  - IsActiveMember
  - age_group
  - tenure_group
  - geo_gender

- Numerical Features:
  - CreditScore
  - Age
  - Tenure
  - Balance
  - NumOfProducts
  - EstimatedSalary
  - balance_salary_ratio
  - high_bal_low_sal
  - credit_util
  - ZeroBalance

### Model Parameters
- SMOTE:
  - k_neighbors: 3
  - sampling_strategy: 0.7

- XGBoost:
  - n_estimators: 300
  - max_depth: 6
  - learning_rate: 0.01
  - subsample: 1.0
  - colsample_bytree: 0.6
  - scale_pos_weight: 2

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- Classification Report

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request