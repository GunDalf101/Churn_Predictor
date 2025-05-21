import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.model import ChurnPredictor
from src.feature_engineering import create_features

st.set_page_config(
    page_title="Banking Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffcdd2;
        border: 1px solid #ef9a9a;
    }
    .low-risk {
        background-color: #c8e6c9;
        border: 1px solid #81c784;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Banking Churn Predictor")
st.markdown("""
    This application predicts the likelihood of a customer churning from the bank.
    Fill in the customer details below to get a prediction.
""")

@st.cache_resource
def load_model():
    return ChurnPredictor.load_model()

try:
    predictor = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 95, 35)
    credit_score = st.slider("Credit Score", 300, 850, 650)

with col2:
    st.subheader("Banking Details")
    tenure = st.slider("Tenure (years)", 0, 10, 2)
    balance = st.number_input("Balance", 0.0, 250000.0, 10000.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.checkbox("Has Credit Card", value=True)
    is_active = st.checkbox("Is Active Member", value=True)
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

input_dict = {
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'CreditScore': credit_score,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': int(has_cr_card),
    'IsActiveMember': int(is_active),
    'EstimatedSalary': estimated_salary
}

input_df = pd.DataFrame([input_dict])

input_df = create_features(input_df)

if st.button("Predict Churn Risk"):
    try:
        prediction = predictor.predict(input_df)[0]
        probability = predictor.predict_proba(input_df)[0][1]
        
        st.markdown("### Prediction Results")
        
        risk_level = "high-risk" if probability > 0.5 else "low-risk"
        risk_text = "High Risk of Churn" if probability > 0.5 else "Low Risk of Churn"
        
        st.markdown(f"""
            <div class="prediction-box {risk_level}">
                <h3>{risk_text}</h3>
                <p>Churn Probability: {probability:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Key Factors")
        st.markdown("""
            The following factors most influence the prediction:
            - Credit Score
            - Age
            - Balance
            - Number of Products
            - Geography
        """)
        
        st.markdown("### Recommendations")
        if probability > 0.5:
            st.markdown("""
                **High Risk Customer - Suggested Actions:**
                - Review customer's product usage
                - Check for any recent complaints
                - Consider offering retention incentives
                - Schedule a customer success call
            """)
        else:
            st.markdown("""
                **Low Risk Customer - Suggested Actions:**
                - Continue regular engagement
                - Monitor for any changes in behavior
                - Consider upselling opportunities
            """)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built by GunDalf</p>
    </div>
""", unsafe_allow_html=True)
