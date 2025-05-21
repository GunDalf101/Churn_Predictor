"""
Data processing module for loading and preprocessing the churn dataset.
"""

# Data loading and preprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .config import (
    DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN
)

def load_data(file_path: str = None) -> pd.DataFrame:
    """Load the churn dataset from the specified path."""
    if file_path is None:
        file_path = DATA_DIR / "bank-churn.csv"
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset by handling missing values and converting data types."""
    df = df.copy()
    
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def prepare_train_test_data(df: pd.DataFrame):
    """Prepare training and testing datasets."""
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )