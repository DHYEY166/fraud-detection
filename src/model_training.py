"""
Model training module for fraud detection.
Handles model training, evaluation, and result visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
import time

def balance_dataset(X_train, y_train):
    """
    Balance the dataset using upsampling of the minority class.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        tuple: Balanced features and target
    """
    # Combine features and target
    train_data = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
    
    # Separate majority and minority classes
    train_majority = train_data[train_data['target'] == 0]
    train_minority = train_data[train_data['target'] == 1]
    
    # Upsample minority class
    train_minority_upsampled = resample(
        train_minority,
        replace=True,
        n_samples=len(train_majority),
        random_state=42
    )
    
    # Combine majority and upsampled minority
    train_balanced = pd.concat([train_majority, train_minority_upsampled])
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features and target
    X_train_balanced = train_balanced.drop('target', axis=1)
    y_train_balanced = train_balanced['target']
    
    return X_train_balanced, y_train_balanced

def get_feature_importance(X_train, y_train):
    """
    Calculate feature importance using Ridge and Lasso regression.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    # Ridge Regression for feature importance
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(ridge.coef_)
    }).sort_values('importance', ascending=False)
    
    # Lasso Regression for feature selection
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    lasso_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(lasso.coef_)
    }).sort_values('importance', ascending=False)
    
    return ridge_importance, lasso_importance

def train_and_evaluate_model(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate a single model.
    
    Args:
        model: The model to train
        model_name (str): Name of the model
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        
    Returns:
        dict: Model performance metrics
    """
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'model_name': model_name,
        'training_time': training_time,
        'validation_accuracy': accuracy_score(y_val, val_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'validation_roc_auc': roc_auc_score(y_val, val_pred),
        'test_roc_auc': roc_auc_score(y_test, test_pred),
        'validation_report': classification_report(y_val, val_pred),
        'test_report': classification_report(y_test, test_pred),
        'validation_confusion_matrix': confusion_matrix(y_val, val_pred),
        'test_confusion_matrix': confusion_matrix(y_test, test_pred)
    }
    
    return results

def train_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train multiple models and evaluate their performance.
    
    Args:
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        
    Returns:
        dict: Results for all models
    """
    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(),
        'LightGBM': LGBMClassifier()
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        results[name] = train_and_evaluate_model(
            model, name,
            X_train_balanced, X_val, X_test,
            y_train_balanced, y_val, y_test
        )
    
    return results

def plot_results(results, ridge_importance):
    """
    Plot model performance comparison and feature importance.
    
    Args:
        results (dict): Model results
        ridge_importance (pd.DataFrame): Feature importance scores
    """
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=ridge_importance.head(15),
        x='importance',
        y='feature'
    )
    plt.title('Top 15 Features by Ridge Regression Importance')
    plt.tight_layout()
    plt.show()
    
    # Plot model performance comparison
    model_metrics = pd.DataFrame({
        'Model': list(results.keys()),
        'Validation Accuracy': [results[model]['validation_accuracy'] for model in results],
        'Test Accuracy': [results[model]['test_accuracy'] for model in results],
        'Validation ROC AUC': [results[model]['validation_roc_auc'] for model in results],
        'Test ROC AUC': [results[model]['test_roc_auc'] for model in results]
    })
    
    plt.figure(figsize=(12, 6))
    model_metrics.set_index('Model')[['Validation ROC AUC', 'Test ROC AUC']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('ROC AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def evaluate_models(results):
    """
    Print detailed evaluation metrics for all models.
    
    Args:
        results (dict): Model results
    """
    for name, result in results.items():
        print(f"\nResults for {name}:")
        print(f"Training Time: {result['training_time']:.2f} seconds")
        print(f"Validation Accuracy: {result['validation_accuracy']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Validation ROC AUC: {result['validation_roc_auc']:.4f}")
        print(f"Test ROC AUC: {result['test_roc_auc']:.4f}")
        print("\nValidation Classification Report:")
        print(result['validation_report'])
        print("\nTest Classification Report:")
        print(result['test_report'])

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import preprocess_data
    
    # Preprocess data
    file_path = "synthetic_fraud_data.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_path)
    
    # Get feature importance
    ridge_importance, lasso_importance = get_feature_importance(X_train, y_train)
    
    # Train models
    results = train_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Evaluate and visualize results
    evaluate_models(results)
    plot_results(results, ridge_importance)
