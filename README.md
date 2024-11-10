# Financial Transaction Fraud Detection

## Overview
This project implements various machine learning models to detect fraudulent financial transactions using a synthetic dataset that mimics real-world transaction patterns. The models include Logistic Regression, Random Forest, Gradient Boosting, and LightGBM, with comprehensive feature engineering and model evaluation.

## Dataset
The dataset is a synthetic financial transaction dataset designed to simulate realistic patterns across multiple categories including retail, grocery, dining, travel, and more. Due to its large size, the dataset is not included in this repository but can be accessed at:
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/ismetsemedov/transactions/data)

### Key Features
- Transaction details (amount, timestamp, merchant category)
- Geographic information (country, city, city size)
- Payment method details (card type, card presence)
- Device and channel information
- Velocity metrics (transaction patterns in the last hour)
- Fraud labels for supervised learning

## Project Structure
```
.
├── README.md
├── notebooks/
│   └── fraud_detection.ipynb
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    └── model_training.py
```

## Features
- Comprehensive data preprocessing and feature engineering
- Implementation of multiple machine learning models
- Feature importance analysis using Ridge and Lasso regression
- Class imbalance handling through upsampling
- Model performance evaluation using various metrics
- Visualization of results

## Models Implemented
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- LightGBM Classifier

## Requirements
```
pandas
numpy
scikit-learn
lightgbm
seaborn
matplotlib
```

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset from the Kaggle link provided above and place it in the appropriate directory.

## Usage
1. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

2. Or run the Python scripts:
```bash
python src/data_preprocessing.py
python src/model_training.py
```

## Model Performance
The project achieves the following performance metrics:
- High accuracy in fraud detection
- Balanced precision and recall
- ROC-AUC scores for different models
- Comprehensive evaluation metrics for model comparison

## Feature Importance
The project includes analysis of the most important features for fraud detection, helping understand key indicators of fraudulent transactions.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by Ismet Semedov on Kaggle
- Inspired by real-world financial fraud detection systems
- Built using state-of-the-art machine learning libraries

## Contact
For any queries regarding this project, please open an issue in the GitHub repository.
