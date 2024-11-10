"""
Data preprocessing module for fraud detection.
Handles data loading, cleaning, feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import ast

def load_data(file_path):
    """
    Load and validate the fraud detection dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe
    """
    essential_columns = [
        'customer_id', 'timestamp', 'merchant_category', 'amount', 'country',
        'city_size', 'card_type', 'card_present', 'device', 'channel',
        'distance_from_home', 'high_risk_merchant', 'transaction_hour',
        'weekend_transaction', 'velocity_last_hour', 'is_fraud'
    ]
    
    df = pd.read_csv(file_path, usecols=essential_columns)
    return df

def process_velocity_features(df):
    """
    Process velocity features from the velocity_last_hour column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with processed velocity features
    """
    df['velocity_last_hour'] = df['velocity_last_hour'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    velocity_df = df['velocity_last_hour'].apply(pd.Series)
    df = pd.concat([df, velocity_df], axis=1)
    df.drop(columns=['velocity_last_hour'], inplace=True)
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using Label and One-Hot encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    # Label encoding for ordinal categories
    le = LabelEncoder()
    for col in ['city_size', 'channel']:
        df[col] = le.fit_transform(df[col])
    
    # One-hot encoding for nominal categories
    nominal_cols = ['merchant_category', 'country', 'card_type', 'device']
    df = pd.get_dummies(
        df,
        columns=nominal_cols,
        drop_first=True
    )
    
    return df

def feature_engineering(df):
    """
    Perform feature engineering on the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    
    # Convert categorical columns
    for col in ['merchant_category', 'country', 'city_size', 'card_type', 'device', 'channel']:
        df[col] = df[col].astype('category')
    
    # Convert boolean columns
    binary_cols = ['card_present', 'high_risk_merchant', 'weekend_transaction', 'is_fraud']
    for col in binary_cols:
        df[col] = df[col].map({
            1: 1, 'yes': 1, 'Yes': 1, 'True': 1, True: 1,
            0: 0, 'no': 0, 'No': 0, 'False': 0, False: 0
        }).astype(int)
    
    return df

def split_and_scale_data(df, target_col='is_fraud'):
    """
    Split data into train, validation, and test sets, and scale features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        
    Returns:
        tuple: Scaled train, validation, and test sets with their targets
    """
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Second split: separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test)

def preprocess_data(file_path):
    """
    Complete preprocessing pipeline for fraud detection data.
    
    Args:
        file_path (str): Path to the raw data file
        
    Returns:
        tuple: Processed and split datasets ready for modeling
    """
    # Load data
    df = load_data(file_path)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Process velocity features
    df = process_velocity_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Split and scale data
    return split_and_scale_data(df)

if __name__ == "__main__":
    # Example usage
    file_path = "synthetic_fraud_data.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_path)
    print("Preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
