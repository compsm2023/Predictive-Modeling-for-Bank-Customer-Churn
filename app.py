import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Churn Risk AI",
    page_icon="🏦",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, feature_names = load_assets()

# ---------------- HEADER ----------------
st.title("🏦 Customer Churn Risk Intelligence")
st.markdown(
    "Adjust customer parameters in the sidebar to calculate real-time churn risk scores."
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Customer Profile")

age = st.sidebar.slider("Age", 18, 92, 40)
balance = st.sidebar.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
products = st.sidebar.slider("Number of Products", 1, 4, 1)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
salary = st.sidebar.number_input("Estimated Salary ($)", 1000.0, 200000.0, 50000.0)

active = st.sidebar.radio(
    "Active Member?", [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

cards = st.sidebar.radio(
    "Has Credit Card?", [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

geo = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# ---------------- FEATURE ENGINEERING ----------------
input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products,
    'HasCrCard': cards,
    'IsActiveMember': active,
    'EstimatedSalary': salary,
    'Balance_Salary_Ratio': balance / salary if salary > 0 else 0,
    'Product_Density': products / (tenure + 1),
    'Engagement_Product_Interact': active * products,
    'Age_Tenure_Interact': age * tenure,
    'Geography_Germany': 1 if geo == 'Germany' else 0,
    'Geography_Spain': 1 if geo == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0
}

input_df = pd.DataFrame([input_dict])

# Ensure same feature order as training
input_df = input_df[feature_names]

# ---------------- SCALING ----------------
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ----------------
prob = model.predict_proba(input_scaled)[0][1]
risk_percent = prob * 100

# ---------------- DISPLAY ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Risk Score")

    if risk_percent > 70:
        st.error(f"🔴 HIGH RISK: {risk_percent:.1f}%")
    elif risk_percent > 30:
        st.warning(f"🟠 MEDIUM RISK: {risk_percent:.1f}%")
    else:
        st.success(f"🟢 LOW RISK: {risk_percent:.1f}%")

    # Safe integer progress (0–100)
    st.progress(max(1, min(100, int(risk_percent))))

    st.caption(f"Model Confidence Score: {prob:.4f}")

with col2:
    st.subheader("Business Recommendation")

    if risk_percent > 70:
        st.write("❌ Immediate retention campaign required.")
        st.write("- Offer loyalty bonuses")
        st.write("- Provide special interest rates")
        st.write("- Assign relationship manager")
    elif risk_percent > 30:
        st.write("⚠ Monitor customer engagement.")
        st.write("- Send personalized offers")
        st.write("- Increase digital engagement")
    else:
        st.write("✅ Maintain standard engagement.")
        st.write("Customer shows strong stability.")

st.divider()

st.info(
    "This AI model uses XGBoost with advanced Feature Engineering "
    "(Product Density, Balance Ratio, and Engagement Interaction)."
)
