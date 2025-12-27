📦 Deployment Guide
Platform

Streamlit Community Cloud (Free Tier)

✅ Prerequisites

Before deployment, ensure the following:

GitHub account (logged in)

Repository is public

streamlit_app.py exists in app/ folder

Trained model and imputer saved in models/

requirements.txt present in root directory

📂 Required Repository Structure
ecommerce-churn-prediction/
├── app/
│   ├── streamlit_app.py
│   └── predict.py
├── models/
│   ├── best_model.pkl
│   └── imputer.pkl
├── requirements.txt
└── README.md

📄 requirements.txt

Ensure the following dependencies are listed:

streamlit==1.28.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
joblib==1.3.0
plotly==5.17.0

🚀 Step-by-Step Deployment
1️⃣ Prepare Repository

Commit and push all final files to GitHub

Verify Streamlit app runs locally:

streamlit run app/streamlit_app.py

2️⃣ Deploy on Streamlit Cloud

Go to 👉 https://share.streamlit.io

Sign in with GitHub

Click “New app”

Select:

Repository: ecommerce-churn-prediction

Branch: main

Main file path: app/streamlit_app.py

Click Deploy

⏳ Initial build takes 2–5 minutes.

🔍 Post-Deployment Checks

After deployment, perform the following validations:

App loads without errors

Single customer prediction works

Batch CSV upload works

Model metrics and visualizations display correctly

No runtime errors in Streamlit logs

🌐 Live Application URL

Deployed Streamlit App:
👉 https://ecommerce-churn-prediction-vinay.streamlit.app/

🧪 Testing Checklist

✔ App loads successfully
✔ Single prediction works
✔ Batch prediction works
✔ All visualizations display
✔ No errors in logs

📌 Notes

Model and imputer are loaded using joblib

Resources are cached using @st.cache_resource

App is stateless and safe for cloud deployment

🟢 Deployment Status

Status: ✅ Successfully deployed
Platform: Streamlit Community Cloud
Cost: Free