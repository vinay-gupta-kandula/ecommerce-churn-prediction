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
    except:
        return None, None

model, imputer = load_assets()

# Features used in training (30 features)
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
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Page:", [
    "1. Home", 
    "2. Exploratory Analysis", 
    "3. Individual Prediction", 
    "4. Batch Prediction", 
    "5. Model Performance"
])

# ======================================================
# PAGE 1: HOME
# ======================================================
if page == "1. Home":
    st.title("📊 E-Commerce Customer Churn Prediction")
    st.image("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?auto=format&fit=crop&q=80&w=1000", use_column_width=True)
    
    st.markdown("""
    ### Project Overview
    This system uses machine learning to identify high-risk customers likely to stop purchasing. 
    By predicting churn before it happens, businesses can deploy targeted retention strategies.

    ### Business Objective
    - **Identify** high-risk customers with >70% recall.
    - **Analyze** key behavioral drivers of churn.
    - **Optimize** marketing spend on customers who actually need incentives.

    ### How to use this Dashboard:
    - Use **Exploratory Analysis** to see trends in the data.
    - Use **Individual Prediction** for a quick check on one customer.
    - Use **Batch Prediction** to process a CSV list of thousands of customers.
    """)

# ======================================================
# PAGE 2: EXPLORATORY ANALYSIS
# ======================================================
elif page == "2. Exploratory Analysis":
    st.header("📈 Data Insights & Trends")
    
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='churn', data=df, palette='viridis', ax=ax)
            st.pyplot(fig)
        
        with c2:
            st.subheader("Monetary Value vs Churn")
            fig, ax = plt.subplots()
            sns.boxplot(x='churn', y='monetary_value', data=df, ax=ax)
            ax.set_ylim(0, 5000) # Zoom in
            st.pyplot(fig)
            
        st.subheader("Feature Correlations")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df[['rfm_total', 'monetary_per_txn', 'tenure_velocity', 'churn']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.error("Processed data file not found. Please run the pipeline first.")

# ======================================================
# PAGE 3: INDIVIDUAL PREDICTION
# ======================================================
elif page == "3. Individual Prediction":
    st.header("🔍 Single Customer Assessment")
    st.info(f"Targeting a Recall of 77% using threshold {OPTIMAL_THRESHOLD}")

    with st.form("single_pred"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                inputs[feat] = st.number_input(feat.replace("_", " ").title(), value=0.0)
        
        btn = st.form_submit_button("Predict Churn")
    
    if btn:
        input_df = pd.DataFrame([inputs])
        X_scaled = imputer.transform(input_df[FEATURE_COLUMNS])
        prob = model.predict_proba(X_scaled)[0, 1]
        
        st.subheader("Prediction Result")
        if prob >= OPTIMAL_THRESHOLD:
            st.error(f"**HIGH RISK** (Probability: {prob:.2%})")
            st.write("💡 Recommendation: Send a high-value discount coupon.")
        else:
            st.success(f"**LOW RISK** (Probability: {prob:.2%})")
            st.write("💡 Recommendation: Standard engagement via newsletter.")

# ======================================================
# PAGE 4: BATCH PREDICTION
# ======================================================
elif page == "4. Batch Prediction":
    st.header("📁 Bulk Customer Scoring")
    st.write("Upload a CSV file to generate churn probabilities for your entire database.")
    
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df_batch = pd.read_csv(file)
        # Prediction
        X_batch = imputer.transform(df_batch[FEATURE_COLUMNS])
        probs = model.predict_proba(X_batch)[:, 1]
        
        df_batch['Churn_Probability'] = probs
        df_batch['Prediction'] = np.where(probs >= OPTIMAL_THRESHOLD, "CHURN", "ACTIVE")
        
        st.write("### Scored Results")
        st.dataframe(df_batch[['Churn_Probability', 'Prediction'] + FEATURE_COLUMNS])
        
        csv = df_batch.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "scored_customers.csv", "text/csv")

# ======================================================
# PAGE 5: MODEL PERFORMANCE
# ======================================================
elif page == "5. Model Performance":
    st.header("⚖️ Model Evaluation Metrics")
    
    if SUBMISSION_PATH.exists():
        with open(SUBMISSION_PATH) as f:
            sub = json.load(f)
            perf = sub["final_model_performance"]["test_set_metrics"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{perf['roc_auc']:.4f}")
        m2.metric("Recall", f"{perf['recall']:.1%}")
        m3.metric("Precision", f"{perf['precision']:.1%}")
        m4.metric("F1 Score", f"{perf['f1_score']:.4f}")

        st.divider()
        st.subheader("Visual Evaluation")
        v_cols = st.columns(2)
        with v_cols[0]:
            st.image(str(BASE_DIR / "visualizations" / "roc_curve.png"), caption="ROC Curve")
        with v_cols[1]:
            st.image(str(BASE_DIR / "visualizations" / "feature_importance.png"), caption="Feature Drivers")
    else:
        st.warning("Submission file not found.")