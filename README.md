# 📊 E-Commerce Customer Churn Prediction

## Project Overview

Customer churn is a major challenge for e-commerce businesses, as retaining existing customers is significantly more cost-effective than acquiring new ones. This project delivers an end-to-end machine learning solution to predict customer churn using historical transactional data.

The solution covers the complete lifecycle: data acquisition, cleaning, feature engineering, exploratory analysis, model training, strict leakage-free evaluation, cross-validation, and final deployment as an interactive Streamlit web application. Special care is taken to preserve temporal integrity, ensuring all reported metrics are realistic and production-ready.

---

## Business Problem

The goal of this project is to identify customers who are at high risk of churn so that the business can take proactive retention actions such as personalized offers, discounts, or engagement campaigns.

The problem is formulated as a binary classification task:

* **1** → Customer churned
* **0** → Customer remains active

Accurate churn prediction enables:

* Reduced revenue loss
* Improved customer lifetime value (CLV)
* Better targeting of marketing resources

---

## Dataset

* **Source:** UCI Machine Learning Repository
[https://archive.ics.uci.edu/ml/datasets/Online+Retail+II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
* **Raw Size:** ~541,909 rows × 8 columns
* **Final Customers:** ~3,223
* **Time Period:** 2009 – 2011

The dataset consists of invoice-level retail transactions, including purchase timestamps, quantities, prices, and customer identifiers.

---

## Methodology

### 1. Data Cleaning

Key preprocessing steps included:

* Removing transactions with missing CustomerID
* Filtering cancelled invoices
* Removing negative quantities and prices
* Deduplicating records
* Standardizing date formats
* Validating transactional consistency

---

### 2. Feature Engineering

Customer-level features were engineered using only historical data before a fixed cutoff date to avoid temporal leakage:

* **RFM Features**
* Recency (days since last purchase)
* Frequency (number of purchases)
* Monetary value (total and average spend)


* **Behavioral Features**
* Average basket size
* Purchase interval statistics
* Product diversity


* **Temporal Features**
* Customer lifetime
* Activity in recent time windows (30/60/90 days)



**Final feature count:** 30 features

---

### 3. Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.652 | 0.516 | 0.733 | 0.606 | **0.731** |
| Decision Tree | 0.612 | 0.480 | 0.764 | 0.589 | 0.685 |
| Random Forest | 0.671 | 0.537 | 0.696 | 0.606 | 0.735 |
| Gradient Boosting | 0.680 | 0.566 | 0.526 | 0.545 | 0.729 |
| Neural Network | 0.621 | 0.482 | 0.537 | 0.508 | 0.654 |

---

### 4. Final Model

* **Selected Model:** Logistic Regression (Tuned)
* **Reason:** Best balance of high recall and model stability for imbalanced data.
* **Optimized Performance (Threshold 0.521):**
* ROC-AUC: **0.7488**
* Precision: **0.53**
* Recall: **0.77**
* F1-Score: **0.6293**



Earlier experiments that produced higher scores were discarded after identifying temporal data leakage. The final metrics reflect realistic, trustworthy performance suitable for deployment.

---

## Installation & Usage

### Local Setup

#### Clone Repository

```bash
git clone https://github.com/vinay-gupta-kandula/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction

```

#### Install Dependencies

```bash
pip install -r requirements.txt

```

#### Run Data Pipeline

```bash
python src/01_data_acquisition.py
python src/02_data_cleaning.py
python src/03_feature_engineering.py
python src/04_model_preparation.py

```

#### Train Models (Jupyter)

```bash
jupyter notebook notebooks/05_advanced_models.ipynb

```

#### Launch Web App

```bash
streamlit run app/streamlit_app.py

```

---

## Live Application

🌐 **Streamlit App URL:** [https://ecommerce-churn-prediction-vinay.streamlit.app/](https://ecommerce-churn-prediction-vinay.streamlit.app/)

---

## Project Structure

```
ecommerce-churn-prediction/
├── app/
│   ├── predict.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── 01_business_problem.md
│   ├── 02_project_scope.md
│   ├── 03_technical_approach.md
│   ├── 04_success_criteria.md
│   ├── 08_churn_definition.md
│   ├── 10_eda_insights.md
│   ├── 13_technical_documentation.md
│   └── 14_self_assessment.md
├── deployment/
│   └── deployment_guide.md
├── models/
│   ├── best_model.pkl
│   └── imputer.pkl
├── notebooks/
│   ├── 01_initial_data_exploration.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_feature_eda.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_cross_validation.ipynb
├── src/
│   ├── 01_data_acquisition.py
│   ├── 02_data_cleaning.py
│   ├── 03_feature_engineering.py
│   └── 04_model_preparation.py
├── visualizations/
│   ├── final_confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   ├── prediction_distribution.png
│   └── calibration_curve.png
├── presentation.pdf
├── requirements.txt
└── README.md

```

---

## Results & Business Impact

* Enables proactive churn prevention
* Supports targeted retention campaigns
* Reduces revenue loss at low incentive cost
* Prioritizes recall (**77%**) to minimize missed churners

**Recommendations:**

* Score customers weekly or monthly
* Target top 20–30% highest-risk customers
* Monitor recall and campaign ROI
* Periodically retrain the model with new data

---

## Deployment

* **Platform:** Streamlit Community Cloud
* **Deployment Guide:** `deployment/deployment_guide.md`
* **Live URL:** [https://ecommerce-churn-prediction-vinay.streamlit.app/](https://ecommerce-churn-prediction-vinay.streamlit.app/)

---

## Presentation

📄 **Project Presentation:** `presentation.pdf` (included in repository)