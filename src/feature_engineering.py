"""
Feature engineering module for the churn prediction project.
"""

import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the churn prediction model.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Zero balance indicator
    df['ZeroBalance'] = (df['Balance'] == 0).astype(int)
    
    # Balance to salary ratio (avoid division by zero)
    df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    
    # Age bins
    df['age_group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], 
                            labels=['18-29', '30-44', '45-59', '60+'])
    
    # Tenure buckets
    df['tenure_group'] = pd.cut(df['Tenure'], bins=[-1, 2, 5, 10, 15], 
                               labels=['0-2', '3-5', '6-10', '10+'])
    
    # High balance but low salary
    balance_threshold = df['Balance'].quantile(0.75)
    salary_threshold = df['EstimatedSalary'].quantile(0.25)
    df['high_bal_low_sal'] = ((df['Balance'] > balance_threshold) & 
                             (df['EstimatedSalary'] < salary_threshold)).astype(int)
    
    # Interaction Geography & Gender
    df['geo_gender'] = df['Geography'] + '_' + df['Gender']
    
    # Credit utilization proxy
    df['credit_util'] = df['Balance'] / (df['CreditScore'] + 1)
    
    # Handle NaN values
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df 