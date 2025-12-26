import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =========================
# PATH SETUP (CRITICAL)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.pkl"
FEATURE_INFO_PATH = BASE_DIR / "data" / "processed" / "feature_info.json"
METRICS_PATH = BASE_DIR / "data" / "processed" / "model_metrics.json"

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_imputer():
    return joblib.load(IMPUTER_PATH)

model = load_model()
imputer = load_imputer()

# =========================
# FEATURE COUNT (SAFE)
# =========================
with open(FEATURE_INFO_PATH, "r") as f:
    feature_info = json.load(f)

NUM_FEATURES = feature_info["total_features"]  # 20
INPUT_FEATURES = NUM_FEATURES - 2  # drop churn + customerid

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Single Prediction",
        "Batch Prediction",
        "Model Dashboard",
        "Documentation"
    ]
)

# =========================
# PAGE 1 — HOME
# =========================
if page == "Home":
    st.title("E-Commerce Customer Churn Prediction")

    st.markdown("""
    ### 📌 Project Overview
    This application predicts whether an e-commerce customer is likely to **churn in the next 120 days**.

    **Model Highlights**
    - Gradient Boosting Classifier
    - ROC-AUC ≈ **0.74**
    - Temporal leakage-safe design
    """)

    st.metric("Total Customers", feature_info["total_customers"])
    st.metric("Churn Rate", round(feature_info["churn_rate"], 3))
    st.metric("Churn Window (days)", feature_info["churn_window_days"])

# =========================
# PAGE 2 — SINGLE PREDICTION
# =========================
elif page == "Single Prediction":
    st.title("🔍 Single Customer Prediction")

    st.info("Enter numeric feature values only (same order as training set).")

    inputs = []
    cols = st.columns(2)

    for i in range(INPUT_FEATURES):
        with cols[i % 2]:
            val = st.number_input(
                f"Feature {i+1}",
                value=0.0
            )
            inputs.append(val)

    if st.button("Predict Churn Risk"):
        X = np.array(inputs).reshape(1, -1)
        X = imputer.transform(X)

        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= 0.5)

        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{prob:.2%}")
        st.metric("Prediction", "CHURN" if pred == 1 else "ACTIVE")

        if prob >= 0.7:
            st.error("⚠️ High churn risk – immediate retention action recommended.")
        elif prob >= 0.4:
            st.warning("⚠️ Medium churn risk – monitor closely.")
        else:
            st.success("✅ Low churn risk.")

# =========================
# PAGE 3 — BATCH PREDICTION
# =========================
elif page == "Batch Prediction":
    st.title("📂 Batch Prediction")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if df.shape[1] != INPUT_FEATURES:
            st.error(f"Expected {INPUT_FEATURES} features, got {df.shape[1]}")
        else:
            X = imputer.transform(df.values)
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)

            df["churn_probability"] = probs
            df["churn_prediction"] = preds

            st.success("Predictions generated successfully")
            st.dataframe(df.head())

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

# =========================
# PAGE 4 — MODEL DASHBOARD
# =========================
elif page == "Model Dashboard":
    st.title("📊 Model Performance")

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", round(metrics["roc_auc"], 3))
    col2.metric("Precision", round(metrics["precision"], 3))
    col3.metric("Recall", round(metrics["recall"], 3))

    st.markdown("### Confusion Matrix (Test Set)")
    cm = np.array([[metrics["accuracy"], 1 - metrics["accuracy"]],
                   [1 - metrics["recall"], metrics["recall"]]])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    st.pyplot(fig)

# =========================
# PAGE 5 — DOCUMENTATION
# =========================
else:
    st.title("📘 Documentation")

    st.markdown("""
    ### How to Use
    1. Navigate using sidebar
    2. Use **Single Prediction** for individual customers
    3. Use **Batch Prediction** for CSV uploads

    ### Notes
    - Model trained with leakage-safe temporal split
    - Features are numerical only
    - Threshold = 0.5

    ### Contact
    **Author:** Vinay Kandula  
    **Project:** E-Commerce Churn Prediction
    """)
