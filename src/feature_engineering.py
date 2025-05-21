# Feature engineering for churn prediction

import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for the churn prediction model."""
    df = df.copy()
    
    df['ZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    
    df['age_group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], 
                            labels=['18-29', '30-44', '45-59', '60+'])
    df['tenure_group'] = pd.cut(df['Tenure'], bins=[-1, 2, 5, 10, 15], 
                               labels=['0-2', '3-5', '6-10', '10+'])
    
    balance_threshold = df['Balance'].quantile(0.75)
    salary_threshold = df['EstimatedSalary'].quantile(0.25)
    df['high_bal_low_sal'] = ((df['Balance'] > balance_threshold) & 
                             (df['EstimatedSalary'] < salary_threshold)).astype(int)
    
    df['geo_gender'] = df['Geography'] + '_' + df['Gender']
    df['credit_util'] = df['Balance'] / (df['CreditScore'] + 1)
    
    # Handle missing values
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode_values = df[col].mode()
        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values[0])
        else:
            # If no mode exists, fill with the first unique value
            df[col] = df[col].fillna(df[col].iloc[0])
    
    return df 