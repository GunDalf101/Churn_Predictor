"""
Configuration module for the churn prediction project.
"""

# Project configuration

import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Create dirs
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw" / "Churn_Modelling.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"

# Features
CATEGORICAL_FEATURES = [
    'Geography',
    'Gender',
    'HasCrCard',
    'IsActiveMember',
    'age_group',
    'tenure_group',
    'geo_gender'
]

NUMERICAL_FEATURES = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary',
    'balance_salary_ratio',
    'high_bal_low_sal',
    'credit_util',
    'ZeroBalance',
]

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'Exited'

# XGBoost params
MODEL_PARAMS = {
    'classifier__n_estimators': [300, 400, 500],
    'classifier__max_depth': [4, 5, 6],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': [0, 0.1, 0.2],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.8, 0.9, 1.0],
    'classifier__scale_pos_weight': [1, 2, 3],
    'smote__k_neighbors': [3, 5],
    'smote__sampling_strategy': [0.7, 0.8]
} 