# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---- Load only the trained model ----
with open("loan_approval_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")
st.markdown("Fill in the applicant details below to predict loan approval:")

# ---- User Inputs ----
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
loan_amount_term = st.selectbox("Loan Amount Term (in months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---- Create Input DataFrame ----
input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}

df_input = pd.DataFrame(input_dict)

# ---- Basic Encoding (since encoders/scaler not saved) ----
df_input_encoded = pd.get_dummies(df_input)

# Match training feature columns if available
expected_cols = getattr(model, "feature_names_in_", df_input_encoded.columns)
for col in expected_cols:
    if col not in df_input_encoded.columns:
        df_input_encoded[col] = 0
df_input_encoded = df_input_encoded[expected_cols]

# ---- Prediction ----
if st.button("Predict Loan Approval"):
    prediction = model.predict(df_input_encoded)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected")
