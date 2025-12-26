"""
Phase 6.1: Model Preparation

Prepares customer-level feature dataset for modeling.
- Removes identifiers
- Selects numerical features only
- Stratified train/validation/test split
- Scales numerical features
- Saves scaler and prepared datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

DATA_PATH = "data/processed/customer_features.csv"
OUTPUT_DIR = "data/processed"
MODEL_DIR = "models"
RANDOM_STATE = 42


def main():
    print("=== PHASE 6.1: MODEL PREPARATION ===")

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    # Target
    y = df["churn"]

    # Drop target & identifier
    X = df.drop(columns=["churn"])
    if "customerid" in X.columns:
        X = X.drop(columns=["customerid"])

    # Select numerical features only
    X = X.select_dtypes(include=[np.number])

    print(f"Numerical feature matrix shape: {X.shape}")
    print("Churn rate:", round(y.mean(), 3))

    # Stratified split: Train (70%) / Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Validation (15%) / Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print("Split sizes:")
    print("Train:", X_train.shape[0])
    print("Validation:", X_val.shape[0])
    print("Test:", X_test.shape[0])

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create directories
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Save scaler
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    # Save prepared datasets
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train_scaled)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val_scaled)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test_scaled)

    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train.values)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val.values)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test.values)

    print("✅ Model preparation completed successfully")
    print("Scaler and datasets saved")


if __name__ == "__main__":
    main()
