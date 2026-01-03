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

# STRICT FEATURE LIST (30 FEATURES)
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
    ### Welcome
    This interactive dashboard predicts the likelihood of customers churning from an e-commerce platform. 
    It is powered by a **Tuned Logistic Regression** model designed to maximize recall for high-impact business decisions.
    
    ### Core Logic
    - **Target:** Customers who make no purchases for 120 days.
    - **Algorithm:** Logistic Regression (Balanced).
    - **Features:** Behavioral RFM metrics + Temporal Ratios.
    """)

# ======================================================
# PAGE 2: DATA INSIGHTS
# ======================================================
elif page == "2. Data Insights":
    st.header("📈 Deep Dive into Training Data")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Customer Balance")
            fig, ax = plt.subplots()
            sns.countplot(x='churn', data=df, palette='viridis', ax=ax)
            st.pyplot(fig)
        with c2:
            st.subheader("Average Spending Patterns")
            fig, ax = plt.subplots()
            sns.boxplot(x='churn', y='monetary_per_txn', data=df, ax=ax)
            ax.set_ylim(0, 1000)
            st.pyplot(fig)
    else:
        st.warning("Data file not found for visualization.")

# ======================================================
# PAGE 3: INDIVIDUAL PREDICTION
# ======================================================
elif page == "3. Individual Prediction":
    st.header("🔍 Manual Customer Assessment")
    with st.form("manual_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                inputs[feat] = st.number_input(feat.replace("_", " ").title(), value=0.0)
        btn = st.form_submit_button("Run Model")
    
    if btn:
        input_df = pd.DataFrame([inputs])
        # Force column order
        X_final = input_df[FEATURE_COLUMNS]
        X_imp = imputer.transform(X_final)
        prob = model.predict_proba(X_imp)[0, 1]
        
        if prob >= OPTIMAL_THRESHOLD:
            st.error(f"High Risk Detected: {prob:.1%}")
        else:
            st.success(f"Low Risk Detected: {prob:.1%}")

# ======================================================
# PAGE 4: BATCH PREDICTION
# ======================================================
elif page == "4. Batch Prediction":
    st.header("📁 Bulk Processing (CSV)")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df_raw = pd.read_csv(file)
        df_raw.columns = df_raw.columns.str.strip() # Clean column names
        
        # Validate all 30 features are present
        missing = [c for c in FEATURE_COLUMNS if c not in df_raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            # Predict using the specified feature list order
            X_batch = df_raw[FEATURE_COLUMNS]
            X_imp = imputer.transform(X_batch)
            probs = model.predict_proba(X_imp)[:, 1]
            
            df_raw['Probability'] = probs
            df_raw['Result'] = np.where(probs >= OPTIMAL_THRESHOLD, "CHURN", "ACTIVE")
            st.dataframe(df_raw[['Result', 'Probability'] + FEATURE_COLUMNS])

# ======================================================
# PAGE 5: PERFORMANCE & DOCS
# ======================================================
else:
    st.header("📘 Documentation & Performance")
    
    # 5.1 Technical Metrics
    if SUBMISSION_PATH.exists():
        with open(SUBMISSION_PATH) as f:
            sub = json.load(f)
            perf = sub["final_model_performance"]["test_set_metrics"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{perf['roc_auc']:.4f}")
        m2.metric("Recall", f"{perf['recall']:.1%}")
        m3.metric("Precision", f"{perf['precision']:.1%}")
        m4.metric("Optimal Threshold", f"{OPTIMAL_THRESHOLD}")
        
    # 5.2 Documentation Text
    st.divider()
    st.markdown(f"""
    ### Feature Glossary
    - **Tenure Velocity:** The rate of purchase relative to customer age.
    - **Variety Ratio:** Unique products divided by total quantity.
    - **Price Stability:** Standard deviation of unit prices across orders.
    
    ### Optimization Logic
    The model is tuned to a probability threshold of **{OPTIMAL_THRESHOLD}**. 
    Lowering this threshold increases **Recall** (capturing more churners) while increasing it improves **Precision** (avoiding false alarms).
    """)
    
    # 5.3 Visuals
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(BASE_DIR / "visualizations" / "roc_curve.png"), caption="ROC Analysis")
    with c2:
        st.image(str(BASE_DIR / "visualizations" / "feature_importance.png"), caption="Key Prediction Drivers")