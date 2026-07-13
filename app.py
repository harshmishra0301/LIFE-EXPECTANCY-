import streamlit as st
import pandas as pd
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("best_model.pkl")

st.set_page_config(
    page_title="AI Insurance Predictor",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3, h4 {color: white;}
.stMetric {background-color: #1c1f26; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("💰 AI Health Insurance Predictor")
st.caption("Smart prediction powered by Machine Learning")

tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Insights", "ℹ️ About"])

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧑 Personal Info")
        age = st.slider("Age", 18, 100, 25)
        gender = st.selectbox("Gender", list(le_gender.classes_))
        children = st.slider("Children", 0, 8, 1)

    with col2:
        st.subheader("🩺 Health Info")
        bmi = st.slider("BMI", 10.0, 60.0, 22.0)
        bloodpressure = st.slider("Blood Pressure", 60, 200, 120)
        diabetic = st.selectbox("Diabetic", list(le_diabetic.classes_))
        smoker = st.selectbox("Smoker", list(le_smoker.classes_))

    currency = st.selectbox(
        "💱 Select Currency",
        ["USD ($)", "INR (₹)", "EUR (€)", "GBP (£)"]
    )

    rates = {
        "USD ($)": 1,
        "INR (₹)": 83,
        "EUR (€)": 0.92,
        "GBP (£)": 0.78
    }

    symbols = {
        "USD ($)": "$",
        "INR (₹)": "₹",
        "EUR (€)": "€",
        "GBP (£)": "£"
    }

    if st.button("🚀 Predict Now"):

        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "bmi": [bmi],
            "bloodpressure": [bloodpressure],
            "diabetic": [diabetic],
            "children": [children],
            "smoker": [smoker]
        })

        input_data["gender"] = le_gender.transform(input_data["gender"])
        input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
        input_data["smoker"] = le_smoker.transform(input_data["smoker"])

        num_cols = ["age", "bmi", "bloodpressure", "children"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        prediction = model.predict(input_data)[0]

        converted_value = prediction * rates[currency]
        symbol = symbols[currency]

        if prediction < 10000:
            risk = "Low Risk 🟢"
        elif prediction < 30000:
            risk = "Medium Risk 🟡"
        else:
            risk = "High Risk 🔴"

        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Predicted Cost", f"{symbol} {converted_value:,.0f}")
        m2.metric("⚠️ Risk Level", risk)
        m3.metric("📊 BMI", bmi)

with tab2:

    st.subheader("📈 Cost Comparison")

    avg_cost = 20000

    chart_data = pd.DataFrame({
        "Category": ["Average Cost", "Your Prediction"],
        "Cost": [avg_cost, converted_value if 'converted_value' in locals() else 0]
    })

    st.bar_chart(chart_data.set_index("Category"))

    st.info("This chart compares your predicted cost with average insurance cost.")

with tab3:

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    This is an AI-powered web application that predicts health insurance costs based on user inputs.

    Tech Stack:
    - Python
    - Streamlit
    - Machine Learning Model

    Features:
    - Real-time prediction
    - Risk analysis
    - Multi-currency support
    - Interactive dashboard

    Built for portfolio & real-world deployment.
    """)

st.markdown("---")
st.caption("🚀 Developed by Ansh | Future Data Scientist")