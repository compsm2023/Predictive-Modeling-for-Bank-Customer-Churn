import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bank Churn Risk AI", layout="wide")

# --- CACHING: This makes the app FAST ---
@st.cache_resource
def load_assets():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

model, scaler, feature_names = load_assets()

# --- UI HEADER ---
st.title("🏦 Customer Churn Risk Intelligence")
st.markdown("Adjust customer parameters in the sidebar to calculate real-time risk scores.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Customer Profile")
age = st.sidebar.slider("Age", 18, 92, 40)
balance = st.sidebar.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
products = st.sidebar.slider("Number of Products", 1, 4, 1)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
salary = st.sidebar.number_input("Estimated Salary ($)", 1000.0, 200000.0, 50000.0)
active = st.sidebar.radio("Active Member?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
cards = st.sidebar.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
geo = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# --- DATA PROCESSING ---
# 1. Create Input DataFrame
input_dict = {
    'CreditScore': credit_score, 'Age': age, 'Tenure': tenure, 'Balance': balance,
    'NumOfProducts': products, 'HasCrCard': cards, 'IsActiveMember': active, 'EstimatedSalary': salary,
    'Balance_Salary_Ratio': balance / salary if salary > 0 else 0,
    'Product_Density': products / (tenure + 1),
    'Engagement_Product_Interact': active * products,
    'Age_Tenure_Interact': age * tenure,
    'Geography_Germany': 1 if geo == 'Germany' else 0,
    'Geography_Spain': 1 if geo == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0
}
input_df = pd.DataFrame([input_dict])

# 2. Scale (Scale only the columns that were scaled during training)
input_scaled = scaler.transform(input_df)

# --- PREDICTION ---
prob = model.predict_proba(input_scaled)[0][1]
risk_percent = prob * 100

# --- DISPLAY RESULTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Risk Score")
    if risk_percent > 70:
        st.error(f"HIGH RISK: {risk_percent:.1f}%")
    elif risk_percent > 30:
        st.warning(f"MEDIUM RISK: {risk_percent:.1f}%")
    else:
        st.success(f"LOW RISK: {risk_percent:.1f}%")
    
    st.progress(prob)

with col2:
    st.subheader("Business Recommendation")
    if risk_percent > 50:
        st.write("❌ **Action:** Trigger retention campaign. Offer personalized interest rates or loyalty bonuses.")
    else:
        st.write("✅ **Action:** Maintain standard engagement. Customer shows high stability.")

st.divider()
st.info("This model uses XGBoost with Feature Engineering (Product Density & Engagement Interaction).")
