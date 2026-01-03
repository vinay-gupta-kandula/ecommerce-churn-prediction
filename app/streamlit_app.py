import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="E-Commerce Customer Churn Prediction",
    layout="wide"
)

# ======================================================
# PATHS (CLOUD + LOCAL SAFE)
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.pkl"
# Path to your updated submission.json for accurate metrics
SUBMISSION_PATH = BASE_DIR / "submission.json" 

# ======================================================
# LOAD ARTIFACTS (CACHED)
# ======================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"❌ model not found at {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_imputer():
    if not IMPUTER_PATH.exists():
        st.error(f"❌ imputer not found at {IMPUTER_PATH}")
        st.stop()
    return joblib.load(IMPUTER_PATH)

model = load_model()
imputer = load_imputer()

# ======================================================
# FEATURE ORDER (UPDATED TO MATCH YOUR 30 FEATURES)
# ======================================================
FEATURE_COLUMNS = [
    "frequency", "monetary_value", "avg_order_value", "total_quantity", 
    "unique_products", "min_unit_price", "max_unit_price", "avg_unit_price", 
    "std_unit_price", "country_count", "customer_tenure_days", "avg_basket_size", 
    "std_basket_size", "max_basket_size", "purchases_last_30_days", 
    "purchases_last_60_days", "purchases_last_90_days", "recency_score", 
    "freq_score", "monetary_score", "rfm_total", "monetary_per_txn", 
    "quantity_per_txn", "tenure_velocity", "variety_ratio", "price_stability", 
    "basket_growth", "log_monetary", "log_frequency", "revenue_per_day"
]

# OPTIMAL THRESHOLD FROM YOUR EVALUATION
OPTIMAL_THRESHOLD = 0.521

# ======================================================
# PREDICTION FUNCTION
# ======================================================
def predict(df):
    # Ensure columns match training order
    X = df[FEATURE_COLUMNS]
    X_imp = imputer.transform(X)
    prob = model.predict_proba(X_imp)[:, 1]
    label = (prob >= OPTIMAL_THRESHOLD).astype(int)
    return label, prob

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Single Prediction", "Batch Prediction", "Model Dashboard", "Documentation"]
)

# ======================================================
# PAGE 1: HOME
# ======================================================
if page == "Home":
    st.title("📊 E-Commerce Customer Churn Prediction")

    st.markdown("""
    This application predicts **customer churn risk** using historical
    transactional behavior and a **Tuned Logistic Regression model**.

    **Key Metrics (Test Set):**
    """)

    if SUBMISSION_PATH.exists():
        with open(SUBMISSION_PATH) as f:
            sub_data = json.load(f)
            metrics = sub_data["final_model_performance"]["test_set_metrics"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        c2.metric("Recall", f"{metrics['recall']:.2%}")
        c3.metric("Precision", f"{metrics['precision']:.2%}")
        c4.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    else:
        st.info("Metrics data available in Model Dashboard.")

# ======================================================
# PAGE 2: SINGLE PREDICTION
# ======================================================
elif page == "Single Prediction":
    st.header("🔍 Single Customer Prediction")
    st.info(f"Using Optimal Decision Threshold: {OPTIMAL_THRESHOLD}")

    inputs = {}
    cols = st.columns(3)

    for i, feature in enumerate(FEATURE_COLUMNS):
        with cols[i % 3]:
            inputs[feature] = st.number_input(feature.replace("_", " ").title(), value=0.0)

    if st.button("Predict Churn Risk"):
        df_input = pd.DataFrame([inputs])
        label, prob = predict(df_input)

        st.subheader("Result")
        st.metric("Churn Probability", f"{prob[0]:.2%}")

        if prob[0] >= 0.7:
            st.error(f"🔴 HIGH RISK (Prob > 70%) — Retention action required.")
        elif prob[0] >= OPTIMAL_THRESHOLD:
            st.warning(f"🟠 MEDIUM RISK (Prob > {OPTIMAL_THRESHOLD*100:.1f}%) — Potential churner.")
        else:
            st.success("🟢 LOW RISK — Customer is likely to remain active.")

# ======================================================
# PAGE 3: BATCH PREDICTION
# ======================================================
elif page == "Batch Prediction":
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV containing the required features for multiple customers.")

    uploaded = st.file_uploader("Upload CSV file", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        missing = set(FEATURE_COLUMNS) - set(df.columns)
        if missing:
            st.error(f"Missing columns in CSV: {list(missing)}")
            st.stop()

        label, prob = predict(df)

        df["churn_probability"] = prob
        df["churn_prediction"] = np.where(prob >= OPTIMAL_THRESHOLD, "CHURN", "ACTIVE")

        st.write("### Prediction Results (Preview)")
        st.dataframe(df[["churn_probability", "churn_prediction"] + FEATURE_COLUMNS].head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Results",
            csv,
            "churn_batch_results.csv",
            "text/csv"
        )

# ======================================================
# PAGE 4: MODEL DASHBOARD
# ======================================================
elif page == "Model Dashboard":
    st.header("📈 Model Performance Summary")

    if SUBMISSION_PATH.exists():
        with open(SUBMISSION_PATH) as f:
            sub_data = json.load(f)
            st.write("### Best Model: Logistic Regression (Tuned)")
            st.json(sub_data["final_model_performance"])
            
            st.write("### Feature Importance")
            st.image("visualizations/feature_importance.png")
    else:
        st.warning("Please ensure visualizations/ and submission.json exist in the project root.")

# ======================================================
# PAGE 5: DOCUMENTATION
# ======================================================
else:
    st.header("📘 Project Documentation")

    st.markdown(f"""
    **Model Architecture**
    - Algorithm: **Logistic Regression** (Balanced Class Weights)
    - Threshold: **{OPTIMAL_THRESHOLD}** (Optimized for F1-Score)
    - Feature Count: **30 Engineered Features**
    
    **Success Criteria**
    - Achieved ROC-AUC: **0.7488**
    - Target Recall: **> 70%**
    - Handling Strategy: Temporal split to prevent data leakage.

    **Author**
    Vinay Gupta Kandula
    """)