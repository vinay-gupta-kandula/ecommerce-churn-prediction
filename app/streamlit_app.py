import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ======================================================
# GLOBAL CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="E-Commerce Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.pkl"
FEATURE_COL_PATH = BASE_DIR / "models" / "feature_columns.pkl"
SUBMISSION_PATH = BASE_DIR / "submission.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"

OPTIMAL_THRESHOLD = 0.521

# ======================================================
# LOAD TRAINED ASSETS
# ======================================================
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    feature_cols = joblib.load(FEATURE_COL_PATH)
    return model, imputer, feature_cols

model, imputer, FEATURE_COLUMNS = load_assets()

if model is None or imputer is None or FEATURE_COLUMNS is None:
    st.error("Model artifacts missing. Please check deployment.")
    st.stop()

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Page:", [
    "1. Home",
    "2. Data Insights",
    "3. Individual Prediction",
    "4. Batch Prediction",
    "5. Performance & Docs"
])

# ======================================================
# PAGE 1: HOME
# ======================================================
if page == "1. Home":
    st.title("📊 E-Commerce Churn Prediction System")
    st.markdown("""
    **Model:** Tuned Logistic Regression  
    **ROC-AUC:** 0.7488  
    **Recall (Optimized):** 77%  
    **Goal:** Identify customers at risk of churn for proactive retention
    """)

# ======================================================
# PAGE 2: DATA INSIGHTS
# ======================================================
elif page == "2. Data Insights":
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            sns.countplot(x="churn", data=df, ax=ax)
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            sns.boxplot(x="churn", y="monetary_per_txn", data=df, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Training dataset not found.")

# ======================================================
# PAGE 3: INDIVIDUAL PREDICTION
# ======================================================
elif page == "3. Individual Prediction":
    st.header("🔍 Individual Customer Prediction")

    with st.form("manual_input"):
        cols = st.columns(3)
        user_input = {}

        for i, col in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                user_input[col] = st.number_input(col, value=0.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([user_input])
        input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        X_imp = imputer.transform(input_df)
        prob = model.predict_proba(X_imp)[0, 1]

        if prob >= OPTIMAL_THRESHOLD:
            st.error(f"🔴 HIGH RISK ({prob:.1%})")
        else:
            st.success(f"🟢 LOW RISK ({prob:.1%})")

# ======================================================
# PAGE 4: BATCH PREDICTION
# ======================================================
elif page == "4. Batch Prediction":
    st.header("📁 Batch Prediction")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df_batch = pd.read_csv(file)
        df_batch.columns = df_batch.columns.str.strip().str.lower()

        X_batch = df_batch.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        X_imp = imputer.transform(X_batch)

        probs = model.predict_proba(X_imp)[:, 1]
        df_batch["Churn_Probability"] = probs
        df_batch["Risk_Level"] = np.where(
            probs >= OPTIMAL_THRESHOLD, "HIGH", "LOW"
        )

        st.dataframe(df_batch.head(50))
        st.download_button(
            "Download Results",
            df_batch.to_csv(index=False).encode("utf-8"),
            "churn_predictions.csv"
        )

# ======================================================
# PAGE 5: PERFORMANCE & DOCS
# ======================================================
else:
    st.header("📘 Model Performance & Documentation")

    if SUBMISSION_PATH.exists():
        with open(SUBMISSION_PATH) as f:
            metrics = json.load(f)["final_model_performance"]["test_set_metrics"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        c2.metric("Recall", f"{metrics['recall']:.1%}")
        c3.metric("Precision", f"{metrics['precision']:.1%}")
        c4.metric("Threshold", OPTIMAL_THRESHOLD)

    st.divider()

    st.subheader("📌 Project Details")
    st.markdown("""
    **Project Title:** E-Commerce Customer Churn Prediction  
    **Model Used:** Logistic Regression (Optimized Threshold)  
    **Dataset Size:** 3,223 customers  
    **Deployment:** Streamlit Cloud  
    **Business Objective:** Early churn detection for proactive retention  
    """)

    st.divider()

    st.subheader("👤 Author")
    st.markdown("""
    **Name:** Vinay Gupta Kandula  
    **Role:** Data Scientist / Machine Learning Engineer  
    **Project Type:** End-to-End ML System (Training → Evaluation → Deployment)
    """)

    st.caption("© 2026 Vinay Gupta Kandula | Churn Prediction System")
