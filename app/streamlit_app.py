import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.pkl"
SUBMISSION_PATH = BASE_DIR / "submission.json"

# ======================================================
# LOAD ARTIFACTS
# ======================================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        return model, imputer
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, imputer = load_assets()

# ======================================================
# CONSTANTS (Matches your final training pipeline)
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

OPTIMAL_THRESHOLD = 0.521

# ======================================================
# APP LOGIC
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Individual", "Batch Prediction", "Documentation"])

if page == "Home":
    st.title("📊 E-Commerce Customer Churn Prediction")
    st.markdown("""
    Welcome! This dashboard provides real-time churn risk assessment for e-commerce customers.
    The underlying engine uses a **Tuned Logistic Regression** model optimized for high recall.
    """)
    
    # Display Latest Metrics from submission.json
    if SUBMISSION_PATH.exists():
        with open(SUBMISSION_PATH) as f:
            data = json.load(f)
            metrics = data["final_model_performance"]["test_set_metrics"]
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        c2.metric("Recall (Sensitivity)", f"{metrics['recall']:.1%}")
        c3.metric("Precision", f"{metrics['precision']:.1%}")
        c4.metric("Optimal Threshold", f"{OPTIMAL_THRESHOLD}")
    
    st.image(str(BASE_DIR / "visualizations" / "roc_curve.png"), caption="Final Model ROC Curve")

elif page == "Predict Individual":
    st.header("🔍 Individual Customer Assessment")
    st.info(f"Threshold: {OPTIMAL_THRESHOLD} | Features: {len(FEATURE_COLUMNS)}")

    with st.form("prediction_form"):
        cols = st.columns(3)
        user_input = {}
        
        for i, feat in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                # Prettify labels (e.g., monetary_per_txn -> Monetary Per Txn)
                label = feat.replace("_", " ").title()
                user_input[feat] = st.number_input(label, value=0.0)
        
        submit = st.form_submit_button("Calculate Risk Score")

    if submit:
        input_df = pd.DataFrame([user_input])
        X_processed = imputer.transform(input_df[FEATURE_COLUMNS])
        
        prob = model.predict_proba(X_processed)[0, 1]
        is_churn = prob >= OPTIMAL_THRESHOLD
        
        st.subheader("Results")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Churn Probability", f"{prob:.2%}")
        
        with col_res2:
            if is_churn:
                st.error("Status: HIGH RISK (Likely to Churn)")
            else:
                st.success("Status: LOW RISK (Likely Active)")
        
        # Visualization of risk
        st.progress(prob)

elif page == "Batch Prediction":
    st.header("📁 Batch Processing")
    st.write("Upload a CSV file with customer behavior data to receive risk scores.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Validation
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
        else:
            X_batch = imputer.transform(df[FEATURE_COLUMNS])
            probs = model.predict_proba(X_batch)[:, 1]
            
            df['Churn_Probability'] = probs
            df['Churn_Risk'] = np.where(probs >= OPTIMAL_THRESHOLD, "High Risk", "Low Risk")
            
            st.write("### Preview of Scored Data")
            st.dataframe(df[['Churn_Probability', 'Churn_Risk'] + FEATURE_COLUMNS].head(20))
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Scored CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")

else:
    st.header("📘 Documentation")
    st.markdown(f"""
    ### Model Overview
    - **Algorithm:** Logistic Regression (with Balanced Class Weights)
    - **Features Used:** 30 (RFM, Temporal, and Behavioral Ratios)
    - **Target:** Probability of 120-day inactivity.
    
    ### How to Interpret Scores
    - **Probability < {OPTIMAL_THRESHOLD}:** Customer shows stable behavior.
    - **Probability >= {OPTIMAL_THRESHOLD}:** Customer shows signs of declining activity. Retention intervention (email, discount) recommended.
    
    ### Developed by:
    Vinay Gupta Kandula
    """)