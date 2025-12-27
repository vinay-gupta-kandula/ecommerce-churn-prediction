# 📦 Deployment Guide

## Platform
**Streamlit Community Cloud (Free Tier)**

---

## ✅ Prerequisites

Before deployment, ensure the following requirements are met:

- GitHub account (logged in)
- Repository is **public**
- `streamlit_app.py` exists inside the `app/` folder
- Trained model and imputer saved inside `models/`
- `requirements.txt` present in the root directory

---

## 📂 Required Repository Structure

```

ecommerce-churn-prediction/
├── app/
│   ├── streamlit_app.py
│   └── predict.py
├── models/
│   ├── best_model.pkl
│   └── imputer.pkl
├── requirements.txt
└── README.md

```

---

## 📄 requirements.txt

Ensure the following dependencies are listed:

```

streamlit==1.28.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
joblib==1.3.0
plotly==5.17.0

````

---

## 🚀 Step-by-Step Deployment

### 1️⃣ Prepare Repository

- Commit and push all final files to GitHub
- Verify the Streamlit app runs locally:

```bash
streamlit run app/streamlit_app.py
````

---

### 2️⃣ Deploy on Streamlit Cloud

1. Go to 👉 [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select:

   * **Repository:** `ecommerce-churn-prediction`
   * **Branch:** `main`
   * **Main file path:** `app/streamlit_app.py`
5. Click **Deploy**

⏳ Initial build may take **2–5 minutes**.

---

## 🔍 Post-Deployment Checks

After deployment, validate the following:

* App loads without errors
* Single customer prediction works
* Batch CSV upload works
* Model metrics and visualizations display correctly
* No runtime errors in Streamlit logs

---

## 🌐 Live Application URL

**Deployed Streamlit App:**
👉 [https://ecommerce-churn-prediction-vinay.streamlit.app/](https://ecommerce-churn-prediction-vinay.streamlit.app/)

---

## 🧪 Testing Checklist

* ✔ App loads successfully
* ✔ Single prediction works
* ✔ Batch prediction works
* ✔ All visualizations display
* ✔ No errors in logs

---

## 📌 Notes

* Model and imputer are loaded using `joblib`
* Resources are cached using `@st.cache_resource`
* Application is stateless and safe for cloud deployment

---

## 🟢 Deployment Status

* **Status:** ✅ Successfully deployed
* **Platform:** Streamlit Community Cloud
* **Cost:** Free


