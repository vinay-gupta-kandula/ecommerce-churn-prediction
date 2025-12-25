# Technical Approach

## Problem Framing
Customer churn prediction is formulated as a **binary classification problem**:
- 1 = Churned
- 0 = Active

Regression is not suitable because the outcome is categorical rather than continuous.

## Feature Engineering Strategy
Transactional data is transformed into customer-level features, including:
- RFM metrics (Recency, Frequency, Monetary value)
- Behavioral patterns (purchase intervals, basket size)
- Temporal features (recent activity windows)
- Product affinity metrics

## Modeling Strategy
Multiple algorithms are evaluated to balance:
- Predictive performance
- Interpretability
- Training time and complexity

Models tested include:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Neural Network

## Evaluation Strategy
- ROC-AUC used as primary metric
- Precision and Recall evaluated for business relevance
- Cross-validation used to assess model stability

## Deployment Strategy
The final model is deployed via a **Streamlit web application** with:
- Single and batch predictions
- Interactive dashboards
- Cloud deployment for stakeholder access
