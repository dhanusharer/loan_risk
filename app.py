import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load Model & Threshold
# -----------------------------
model = joblib.load("models/model_v1.pkl")

with open("models/threshold.json", "r") as f:
    threshold = json.load(f)["best_threshold"]

st.set_page_config(page_title="Loan Risk Intelligence System", layout="wide")

st.title("🏦 Loan Risk Intelligence System")
st.markdown("Advanced ML-powered Loan Default Risk Evaluation")

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------

st.sidebar.header("Applicant Information")

person_age = st.sidebar.number_input("Age", 18, 100, 30)
person_income = st.sidebar.number_input("Annual Income", 1000.0, 1_000_000.0, 50000.0)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
person_emp_exp = st.sidebar.number_input("Employment Experience (Years)", 0.0, 50.0, 5.0)
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E"])
loan_amnt = st.sidebar.number_input("Loan Amount", 1000.0, 1_000_000.0, 10000.0)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", 0.0, 50.0, 10.0)
cb_person_default_on_file = st.sidebar.selectbox("Previous Default", ["Y", "N"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length", 0.0, 30.0, 5.0)
previous_loan_defaults_on_file = st.sidebar.selectbox("Past Loan Defaults", ["Y", "N"])
credit_score = st.sidebar.number_input("Credit Score", 300.0, 900.0, 650.0)
person_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
person_education = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master"])

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------

if st.button("🚀 Evaluate Loan Risk"):

    if person_income == 0:
        st.error("Income cannot be zero.")
        st.stop()

    loan_percent_income = loan_amnt / person_income

    # -------------------------------------------------
    # Business Rule Override Layer
    # -------------------------------------------------
    if loan_percent_income > 20:
        st.error("🚫 AUTO REJECTION: Loan burden extremely high (Business Rule)")
        st.stop()

    # Create DataFrame
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_exp": person_emp_exp,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
        "credit_score": credit_score,
        "person_gender": person_gender,
        "person_education": person_education
    }])

    probability = model.predict_proba(input_data)[0][1]

    # -------------------------------------------------
    # Risk Interpretation Layer
    # -------------------------------------------------

    if probability < 0.2:
        risk_label = "🟢 Low Risk"
    elif probability < threshold:
        risk_label = "🟡 Moderate Risk"
    elif probability < 0.7:
        risk_label = "🟠 High Risk"
    else:
        risk_label = "🔴 Severe Risk"

    prediction = probability >= threshold

    st.subheader("📊 Risk Assessment Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability of Default", f"{probability:.2%}")
        st.metric("Threshold", f"{threshold:.2%}")

    with col2:
        st.write("### Risk Category")
        st.write(risk_label)

        if prediction:
            st.error("❌ Final Decision: Loan Rejected")
        else:
            st.success("✅ Final Decision: Loan Approved")

    # -------------------------------------------------
    # Risk Visualization
    # -------------------------------------------------

    st.subheader("📈 Risk Gauge")

    fig, ax = plt.subplots(figsize=(8,2))
    ax.barh(["Risk"], [probability])
    ax.axvline(threshold, linestyle="--")
    ax.set_xlim(0,1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

    # -------------------------------------------------
    # Loan Burden Analysis
    # -------------------------------------------------

    st.subheader("💰 Loan Burden Analysis")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Income", f"{person_income:,.0f}")
        st.metric("Loan Amount", f"{loan_amnt:,.0f}")

    with col4:
        st.metric("Loan / Income Ratio", f"{loan_percent_income:.2f}")

    fig2, ax2 = plt.subplots()
    ax2.scatter(person_income, loan_amnt)
    ax2.set_xlabel("Income")
    ax2.set_ylabel("Loan Amount")
    st.pyplot(fig2)

    # -------------------------------------------------
    # Debug Section (Optional)
    # -------------------------------------------------

    with st.expander("🔍 Debug View"):
        st.write(input_data)
