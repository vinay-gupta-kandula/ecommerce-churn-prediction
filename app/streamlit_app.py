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
# 1. GLOBAL CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="E-Commerce Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.pkl"
SUBMISSION_PATH = BASE_DIR / "submission.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"

# Load Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        return model, imputer
    except Exception as e:
        return None, None

model, imputer = load_assets()

# THE EXACT 30 FEATURES IN THE ORDER YOUR MODEL EXPECTS
# THE FINAL CALIBRATED FEATURE LIST
FEATURE_COLUMNS = [
    "frequency", "total_quantity", "max_unit_price", "country_count",
    "std_basket_size", "purchases_last_60_days", "freq_score", "monetary_per_txn",
    "variety_ratio", "monetary_value", "unique_products",
    "customer_tenure_days", "max_basket_size", "purchases_last_90_days",
    "monetary_score", "quantity_per_txn", "price_stability", "log_frequency",
    "recency_score", "rfm_total", "tenure_velocity", "basket_growth",
    "avg_basket_size_log", "avg_order_value_log", "avg_unit_price_log", 
    "basket_growth_log", "country_count_log", "monetary_value_log", 
    "total_quantity_log", "purchases_last_30_days"
]

OPTIMAL_THRESHOLD = 0.521

# Ensure the rest of your Page 4 code stays like this:
if file is not None:
    df_batch = pd.read_csv(file)
    df_batch.columns = df_batch.columns.str.strip().str.lower()
    
    missing = [c for c in FEATURE_COLUMNS if c not in df_batch.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
    else:
        X_batch = df_batch[FEATURE_COLUMNS]
        X_imp = imputer.transform(X_batch)
        # ... predict ...



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
    st.markdown(f"""
    ### Project Overview
    This system identifies high-risk customers likely to churn (120 days of inactivity) using a **Tuned Logistic Regression** model.
    
    **Current Model Status:** - **ROC-AUC:** 0.7488
    - **Recall:** 77% (Optimized for proactive retention)
    """)

# ======================================================
# PAGE 2: DATA INSIGHTS
# ======================================================
elif page == "2. Data Insights":
    st.header("📈 Training Data Distribution")
    if DATA_PATH.exists():
        df_eda = pd.read_csv(DATA_PATH)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Churn vs Active Balance")
            fig, ax = plt.subplots()
            sns.countplot(x='churn', data=df_eda, palette='viridis', ax=ax)
            st.pyplot(fig)
        with c2:
            st.subheader("Spending behavior (Monetary Per Txn)")
            fig, ax = plt.subplots()
            sns.boxplot(x='churn', y='monetary_per_txn', data=df_eda, ax=ax)
            ax.set_ylim(0, 1000)
            st.pyplot(fig)
    else:
        st.warning("Data file not found at 'data/processed/customer_features.csv'.")

# ======================================================
# PAGE 3: INDIVIDUAL PREDICTION
# ======================================================
elif page == "3. Individual Prediction":
    st.header("🔍 Single Customer Assessment")
    with st.form("manual_entry"):
        cols = st.columns(3)
        user_input = {}
        for i, feat in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                user_input[feat] = st.number_input(feat.replace("_", " ").title(), value=0.0)
        submit = st.form_submit_button("Run Assessment")
    
    if submit:
        # Convert to DF and force order to match imputer
        input_df = pd.DataFrame([user_input])[FEATURE_COLUMNS]
        if imputer:
            X_imp = imputer.transform(input_df)
            prob = model.predict_proba(X_imp)[0, 1]
            
            st.subheader("Assessment Result")
            if prob >= OPTIMAL_THRESHOLD:
                st.error(f"🔴 HIGH RISK DETECTED ({prob:.1%})")
            else:
                st.success(f"🟢 LOW RISK DETECTED ({prob:.1%})")
        else:
            st.error("Model artifacts missing.")

# ======================================================
# PAGE 4: BATCH PREDICTION
# ======================================================
elif page == "4. Batch Prediction":
    st.header("📁 Bulk Processing (CSV Upload)")
    st.write("Upload a CSV file. The app will automatically clean and reorder columns.")
    
    # FIX: Defined 'file' here to prevent NameError
    file = st.file_uploader("Choose CSV File", type="csv")
    
    if file is not None:
        df_batch = pd.read_csv(file)
        # Clean columns: strip spaces and convert to lowercase
        df_batch.columns = df_batch.columns.str.strip().str.lower()
        
        # Check if any columns are missing
        missing = [c for c in FEATURE_COLUMNS if c not in df_batch.columns]
        
        if missing:
            st.error(f"❌ Missing columns in CSV: {missing}")
        else:
            # FIX: Reorder columns to match imputer exactly to prevent ValueError
            X_batch = df_batch[FEATURE_COLUMNS]
            if imputer and model:
                X_imp = imputer.transform(X_batch)
                probs = model.predict_proba(X_imp)[:, 1]
                
                df_batch['Churn_Probability'] = probs
                df_batch['Risk_Level'] = np.where(probs >= OPTIMAL_THRESHOLD, "HIGH", "LOW")
                
                st.write("### Prediction Results (Top 50)")
                st.dataframe(df_batch[['Risk_Level', 'Churn_Probability'] + FEATURE_COLUMNS].head(50))
                
                csv_output = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", data=csv_output, file_name="churn_results.csv")
            else:
                st.error("Model or Imputer not loaded.")

# ======================================================
# PAGE 5: PERFORMANCE & DOCS
# ======================================================
else:
    st.header("📘 Documentation & Evaluation")
    
    if SUBMISSION_PATH.exists():
        with open(SUBMISSION_PATH) as f:
            sub_json = json.load(f)
            metrics = sub_json["final_model_performance"]["test_set_metrics"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        m2.metric("Recall", f"{metrics['recall']:.1%}")
        m3.metric("Precision", f"{metrics['precision']:.1%}")
        m4.metric("Threshold", f"{OPTIMAL_THRESHOLD}")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### Feature Glossary
        - **Revenue Per Day:** Spent / (Tenure + 1).
        - **Variety Ratio:** Unique SKU Count / Total Quantity.
        - **Price Stability:** SD of prices across orders.
        """)
    with c2:
        st.markdown(f"""
        ### Prediction Logic
        We use a probability threshold of **{OPTIMAL_THRESHOLD}**. 
        If $P(Churn) \geq {OPTIMAL_THRESHOLD}$, the customer is flagged for high-priority retention.
        """)

    # Visualizations
    v1, v2 = st.columns(2)
    roc_img = BASE_DIR / "visualizations" / "roc_curve.png"
    fi_img = BASE_DIR / "visualizations" / "feature_importance.png"
    
    if roc_img.exists(): v1.image(str(roc_img), caption="ROC Curve")
    if fi_img.exists(): v2.image(str(fi_img), caption="Feature Drivers")