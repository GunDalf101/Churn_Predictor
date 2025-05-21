# Churn Predictor

A machine learning project for predicting customer churn in banking services using XGBoost and SMOTE. This project implements a comprehensive pipeline for data processing, feature engineering, and model training to predict customer churn with high accuracy.

## 📌 Project Overview

This project aims to predict customer churn in banking services by analyzing customer behavior patterns and demographic information. The model helps identify customers at risk of leaving, enabling proactive retention strategies.

## 🧠 Business Understanding & Problem Statement

Customer churn is a critical issue in banking, where losing customers directly impacts revenue and market share. The problem involves:
- Predicting which customers are likely to leave the bank
- Understanding key factors driving customer churn
- Enabling targeted retention strategies
- Reducing customer acquisition costs

## 📊 Dataset Description

The dataset contains customer information including:
- Demographic data (Age, Gender, Geography)
- Banking relationship details (Tenure, Balance)
- Product usage (NumOfProducts, HasCrCard)
- Financial indicators (CreditScore, EstimatedSalary)
- Customer activity (IsActiveMember)

## 🔍 EDA Summary

Key insights from exploratory data analysis:
- Class imbalance: ~20% churn rate
- Geography impact: Higher churn in certain regions
- Age correlation: Younger customers more likely to churn
- Balance patterns: Zero balance customers show different churn behavior
- Product usage: Customers with more products less likely to churn

## 🛠️ Preprocessing & Feature Engineering

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

## 🤖 Model Choice & Justification

### XGBoost Selection
- Handles non-linear relationships
- Robust to outliers
- Feature importance insights
- Efficient with large datasets
- Built-in handling of missing values

### SMOTE Implementation
- Addresses class imbalance
- Improves minority class prediction
- Maintains data distribution characteristics
- Performs pretty good results when paired with weighted XGBoost.

## 🧪 Evaluation Metrics & Insights

The model is evaluated using:
- Accuracy: Overall prediction correctness
- Precision: True positive rate among predicted positives
- Recall: True positive rate among actual positives
- F1 Score: Harmonic mean of precision and recall
- ROC AUC: Model's ability to distinguish classes
- Confusion Matrix: Detailed prediction breakdown

## 📈 Visuals: Confusion Matrix, ROC, Feature Importance

Key visualizations include:
- Confusion matrix showing true/false predictions
- ROC curve for model performance
- Feature importance plot
- Distribution of predicted probabilities
- Correlation heatmap of features

## 🧬 Handling Imbalance

### SMOTE Implementation
- k_neighbors: 3
- sampling_strategy: 0.7
- Synthetic sample generation
- Balanced class distribution

## 💭 Lessons Learned / Challenges

1. Data Quality
   - Importance of proper feature engineering
   - Handling missing values effectively
   - Dealing with class imbalance

2. Model Development
   - Feature selection impact
   - Hyperparameter tuning importance
   - Balance between complexity and performance

3. Business Impact
   - Cost of false positives vs. false negatives
   - Actionable insights from feature importance
   - Model interpretability needs

## 🚀 Potential Improvements / Next Steps

1. Model Enhancements
   - Ensemble methods exploration
   - Deep learning approaches
   - Automated hyperparameter tuning

2. Feature Engineering
   - Additional interaction features
   - Time-based features
   - Customer behavior patterns

3. Deployment
   - API development
   - Real-time prediction pipeline
   - Monitoring system

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