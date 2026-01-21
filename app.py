import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt  # For chart

# Page config
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval probability.")

# Load model
model = joblib.load("loan_model.pkl")

# ------------------- User Inputs -------------------
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        income = st.number_input("Annual Income (₹)", min_value=0, step=10000)

    with col2:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=10000)
        loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=5)
        cibil = st.slider("CIBIL Score", 300, 900, 700)
        res_asset = st.number_input("Residential Assets Value (₹)", min_value=0, step=50000)
        com_asset = st.number_input("Commercial Assets Value (₹)", min_value=0, step=50000)

    submit = st.form_submit_button("Predict Loan Approval")

# ------------------- Prediction -------------------
if submit:
    # Encode categorical inputs
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0

    input_data = np.array([[dependents, education, self_employed,
                            income, loan_amount, loan_term,
                            cibil, res_asset, com_asset]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Show result
    if prediction == 1:
        st.success(f"Loan Approved (Probability: {probability:.2%})")
    else:
        st.error(f"Loan Rejected (Approval Probability: {probability:.2%})")

    # ------------------- Probability Chart -------------------
    chart_data = pd.DataFrame({
        "Status": ["Approved", "Rejected"],
        "Probability": [probability, 1 - probability]
    })

    chart = alt.Chart(chart_data).mark_bar(color="#4CAF50").encode(
        x='Status',
        y='Probability',
        tooltip=['Status', alt.Tooltip('Probability', format=".2%")]
    ).properties(width=400, height=300, title="Loan Approval Probability")

    st.altair_chart(chart, use_container_width=True)

st.caption("⚠️ This prediction is approximate.")
