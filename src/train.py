"""
Main training script for the churn prediction model.
"""

import pandas as pd
from pathlib import Path

from .data_processing import (
    load_data,
    preprocess_data,
    prepare_train_test_data
)
from .feature_engineering import create_features
from .model import ChurnPredictor
from .utils import plot_feature_importance, plot_confusion_matrix, plot_roc_curve, print_metrics
from .config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df = preprocess_data(df)
    
    # Create engineered features
    print("\nCreating engineered features...")
    df = create_features(df)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    # Initialize and train model
    print("\nTraining model...")
    model = ChurnPredictor()
    model.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print_metrics(metrics)
    
    # Plot results
    print("\nGenerating plots...")
    feature_names = NUMERICAL_FEATURES + [
        f"{col}_{val}" for col in CATEGORICAL_FEATURES 
        for val in df[col].unique()
    ]
    plot_feature_importance(model.model.named_steps['classifier'], feature_names)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    
    # Save model
    print("\nSaving model...")
    model.save_model()
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 