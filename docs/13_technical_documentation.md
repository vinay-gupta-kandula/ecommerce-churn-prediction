# Technical Documentation

## System Architecture
The pipeline is modular, reproducible, and leakage-safe, with strict temporal separation between training and evaluation.

The system follows an end-to-end ML pipeline:
Data acquisition → cleaning → feature engineering → modeling → evaluation → deployment.

## Data Pipeline
- Phase 2: Data acquired from UCI Online Retail II dataset
- Phase 3: Cleaning handled cancellations, missing IDs, outliers
- Phase 4: Customer-level aggregation and churn definition
- Phase 5: EDA performed with statistical validation
- Phase 6–7: Modeling, evaluation, and cross-validation

(See supporting docs:
- 1_business_problem.md
- 8_churn_definition.md
- 10_eda_insights.md)

## Model Architecture
Selected Model: Gradient Boosting Classifier  
Reason: Best balance of ROC-AUC and stability across folds.
Gradient Boosting captures non-linear relationships in customer behavior while remaining interpretable through feature importance.


## API Reference
Prediction API implemented in `app/predict.py`:
- load_model()
- load_scaler()
- preprocess_input()
- predict()
- predict_proba()

## Deployment Architecture
- Streamlit Cloud (free tier)
- Model loaded via joblib
- Stateless UI with cached resources

## Troubleshooting
Common issues:
- NaN values handled via median imputation
- Feature mismatch resolved using fixed feature order
- File path issues avoided using relative paths
