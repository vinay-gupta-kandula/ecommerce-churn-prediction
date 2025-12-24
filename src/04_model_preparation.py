import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import json

# =========================
# Paths
# =========================
DATA_PATH = Path("data/processed/customer_features.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/processed")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test):
    """Return JSON-safe evaluation metrics"""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
        "precision": float(round(precision_score(y_test, y_pred), 4)),
        "recall": float(round(recall_score(y_test, y_pred), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred), 4))
    }


def main():
    print("Starting Phase 5: Model Training & Evaluation...")

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(DATA_PATH)

    y = df["churn"]
    X = df.drop(columns=["customerid", "churn"])

    # Encode (safe even if no categoricals)
    X = pd.get_dummies(X, drop_first=True)

    # -------------------------
    # Train/Test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    # -------------------------
    # Scaling
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # -------------------------
    # Models
    # -------------------------
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    # -------------------------
    # Train & evaluate
    # -------------------------
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        results[name] = metrics
        print(f"{name} metrics:", metrics)

    # -------------------------
    # Select best model
    # -------------------------
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models[best_model_name]

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    # -------------------------
    # Save metrics (JSON-safe)
    # -------------------------
    model_metrics = {
        "best_model": best_model_name,
        "evaluation_metrics": results[best_model_name],
        "all_models": results,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "churn_rate_train": float(round(y_train.mean() * 100, 2)),
        "churn_rate_test": float(round(y_test.mean() * 100, 2))
    }

    with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
        json.dump(model_metrics, f, indent=4)

    print("\nPhase 5 completed successfully ✅")
    print(f"Best model: {best_model_name}")
    print("Best model metrics:", results[best_model_name])


if __name__ == "__main__":
    main()
