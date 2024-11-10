"""
Fraud Detection Package initialization.
This package provides tools for preprocessing financial transaction data
and training fraud detection models.
"""

from .data_preprocessing import preprocess_data, load_data, feature_engineering
from .model_training import train_models, evaluate_models, plot_results

__version__ = '1.0.0'
__author__ = 'Dhyey'
