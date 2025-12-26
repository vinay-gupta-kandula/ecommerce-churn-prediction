"""
Prediction Module for Customer Churn

This module provides a production-ready API for loading models,
preprocessing input data, and generating churn predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "best_model.pkl")
IMPUTER_PATH = os.path.join("models", "imputer.pkl")


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
def load_model():
    """
    Load trained churn prediction model.

    Returns:
        sklearn model
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")

    return joblib.load(MODEL_PATH)


def load_imputer():
    """
    Load feature imputer.

    Returns:
        sklearn SimpleImputer
    """
    if not os.path.exists(IMPUTER_PATH):
        raise FileNotFoundError("Imputer file not found.")

    return joblib.load(IMPUTER_PATH)


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def preprocess_input(input_data):
    """
    Preprocess input data for prediction.

    Supports:
    - Single customer (dict)
    - Batch input (DataFrame)

    Args:
        input_data (dict or pd.DataFrame)

    Returns:
        np.ndarray: processed feature matrix
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input must be dict or pandas DataFrame")

    imputer = load_imputer()
    X = imputer.transform(df)

    return X


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
def predict(input_data):
    """
    Predict churn class (0 or 1).

    Returns:
        list[int]
    """
    model = load_model()
    X = preprocess_input(input_data)
    return model.predict(X).tolist()


def predict_proba(input_data):
    """
    Predict churn probability.

    Returns:
        list[float]
    """
    model = load_model()
    X = preprocess_input(input_data)
    return model.predict_proba(X)[:, 1].tolist()
