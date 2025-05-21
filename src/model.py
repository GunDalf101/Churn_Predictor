# Churn prediction model using XGBoost and SMOTE

import joblib
import numpy as np
import xgboost as xgb
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from .config import MODELS_DIR, RANDOM_STATE

class ChurnPredictor:
    def __init__(self):
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        self.pipeline = Pipeline([
            ('smote', SMOTE(k_neighbors=3, sampling_strategy=0.7, random_state=RANDOM_STATE)),
            ('classifier', xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.01,
                subsample=1.0,
                colsample_bytree=0.6,
                scale_pos_weight=2,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                seed=RANDOM_STATE
            ))
        ])
        self.model = None
        self.feature_columns = None

    def train(self, X_train, y_train):
        # Get dummies for categorical features
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        self.feature_columns = X_train_encoded.columns.tolist()
        self.pipeline.fit(X_train_encoded, y_train)
        self.model = self.pipeline

    def _preprocess_input(self, X):
        # Match training data format
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        for col in self.feature_columns:
            if col not in X_encoded:
                X_encoded[col] = 0
        return X_encoded[self.feature_columns]

    def predict(self, X):
        X_processed = self._preprocess_input(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X):
        X_processed = self._preprocess_input(X)
        return self.model.predict_proba(X_processed)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        return metrics

    def save_model(self, model_path: str = None):
        if model_path is None:
            model_path = MODELS_DIR / "churn_predictor.joblib"
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, model_path)

    @classmethod
    def load_model(cls, model_path: str = None):
        if model_path is None:
            model_path = MODELS_DIR / "churn_predictor.joblib"
        model_data = joblib.load(model_path)
        predictor = cls()
        predictor.model = model_data['model']
        predictor.feature_columns = model_data['feature_columns']
        return predictor 