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

# THE EXACT 30 FEATURES IN THE ORDER YOUR MODEL EXPECTS
FEATURE_COLUMNS = [
    "frequency", "total_quantity", "max_unit_price", "country_count",
    "std_basket_size", "purchases_last_60_days", "freq_score", "monetary_per_txn",
    "variety_ratio", "log_monetary", "monetary_value", "unique_products",
    "avg_unit_price", "customer_tenure_days", "max_basket_size", "purchases_last_90_days",
    "monetary_score", "quantity_per_txn", "price_stability", "log_frequency",
    "avg_order_value", "min_unit_price", "std_unit_price", "avg_basket_size",
    "purchases_last_30_days", "recency_score", "rfm_total", "tenure_velocity",
    "basket_growth", "revenue_per_day"
]

OPTIMAL_THRESHOLD = 0.521

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
    ### Project Overview
    This system uses machine learning to identify high-risk customers likely to stop purchasing. 
    By predicting churn before it happens, businesses can deploy targeted retention strategies.
    
    ### Core Model
    - **Algorithm:** Tuned Logistic Regression
    - **Optimization:** Balanced Class Weights
    - **Target:** Probability of 120-day inactivity.
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
        # Convert to DF and reorder columns
        input_df = pd.DataFrame([user_input])[FEATURE_COLUMNS]
        X_imp = imputer.transform(input_df)
        prob = model.predict_proba(X_imp)[0, 1]
        
        st.subheader("Assessment Result")
        if prob >= OPTIMAL_THRESHOLD:
            st.error(f"🔴 HIGH RISK DETECTED ({prob:.1%})")
        else:
            st.success(f"🟢 LOW RISK DETECTED ({prob:.1%})")

# ======================================================
# PAGE 4: BATCH PREDICTION
# ======================================================
elif page == "4. Batch Prediction":
    st.header("📁 Bulk Processing (CSV Upload)")
    st.write("Upload a CSV file. The app will automatically clean and reorder columns.")
    
    # SOLVED NAME ERROR: Assigned variable 'file' here
    file = st.file_uploader("Choose CSV File", type="csv")
    
    if file:
        df_batch = pd.read_csv(file)
        # Clean columns: strip spaces and convert to lowercase
        df_batch.columns = df_batch.columns.str.strip().str.lower()
        
        # Check if any columns are missing
        missing = [c for c in FEATURE_COLUMNS if c not in df_batch.columns]
        
        if missing:
            st.error(f"❌ Missing columns in CSV: {missing}")
        else:
            # SOLVED VALUE ERROR: Reorder columns to match imputer
            X_batch = df_batch[FEATURE_COLUMNS]
            X_imp = imputer.transform(X_batch)
            probs = model.predict_proba(X_imp)[:, 1]
            
            df_batch['Churn_Probability'] = probs
            df_batch['Risk_Level'] = np.where(probs >= OPTIMAL_THRESHOLD, "HIGH", "LOW")
            
            st.write("### Prediction Results (Top 50)")
            st.dataframe(df_batch[['Risk_Level', 'Churn_Probability'] + FEATURE_COLUMNS].head(50))
            
            csv_output = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", data=csv_output, file_name="churn_results.csv")

# ======================================================
# PAGE 5: PERFORMANCE & DOCS
# ======================================================
else:
    st.header("📘 Documentation & Technical Evaluation")
    
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
    st.markdown("""
    ### Technical Glossary
    - **Revenue Per Day:** Total spent divided by (Customer Tenure + 1).
    - **Variety Ratio:** Count of unique products / Total quantity of items.
    - **Tenure Velocity:** Rate of engagement over the customer's lifespan.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(BASE_DIR / "visualizations" / "roc_curve.png"), caption="ROC Curve Analysis")
    with c2:
        st.image(str(BASE_DIR / "visualizations" / "feature_importance.png"), caption="Top Predictive Features")