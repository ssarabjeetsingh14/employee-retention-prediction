import streamlit as st
import pandas as pd
import pickle
import os

# =====================================================
# Page Configuration (MUST BE FIRST)
# =====================================================
st.set_page_config(
    page_title="Employee Retention Prediction",
    layout="wide"
)

# =====================================================
# Custom Theme CSS (DEPLOYMENT SAFE)
# =====================================================
st.markdown("""
<style>
html, body, .main {
    background-color: #0B0B0B;
    color: #FFFFFF;
}

header {visibility: hidden;}
footer {visibility: hidden;}

.custom-header {
    background: linear-gradient(90deg, #2F4FDB, #3B6FEA, #2F86FF);
    padding: 22px 30px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(47, 79, 219, 0.45);
}

section[data-testid="stSidebar"] {
    background-color: #0F0F0F;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #F9FAFB !important;
    font-weight: 600;
}

input, textarea, select {
    background-color: #141414 !important;
    color: #FFFFFF !important;
    border-radius: 8px;
    border: 1px solid #2A2A2A;
}

.stButton > button {
    background: linear-gradient(90deg, #2F4FDB, #2F86FF);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2B45C7, #256FE0);
}

.result-card {
    background-color: #101010;
    padding: 25px;
    border-radius: 14px;
    border-left: 6px solid #2F86FF;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Header
# =====================================================
st.markdown(
    '<div class="custom-header">Capstone Project | Employee Retention Prediction</div>',
    unsafe_allow_html=True
)

# =====================================================
# Load Model (DEPLOYMENT SAFE – FIXED)
# =====================================================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "pipeline.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("pipeline.pkl not found. Make sure it is committed to GitHub.")

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# =====================================================
# Sidebar Inputs
# =====================================================
st.sidebar.header("Employee Details")

experience = st.sidebar.slider("Experience (Years)", 0, 30, 5)
salary = st.sidebar.number_input("Monthly Salary", min_value=1000, value=30000, step=1000)
training_hours = st.sidebar.slider("Training Hours", 0, 100, 20)
work_life_balance = st.sidebar.selectbox(
    "Work Life Balance",
    ["Low", "Medium", "High"]
)

# =====================================================
# Input Mapping (MATCH TRAINING PIPELINE EXACTLY)
# =====================================================
input_data = pd.DataFrame([{
    "experience": experience,
    "salary": salary,
    "training_hours": training_hours,
    "work_life_balance": work_life_balance
}])

# =====================================================
# Prediction
# =====================================================
if st.button("Predict Retention"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result_text = "Employee Will Stay" if prediction == 0 else "Employee Will Leave"

    st.markdown(
        f"""
        <div class="result-card">
            <b>Prediction:</b> {result_text}<br><br>
            <b>Probability of Leaving:</b> {probability:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )
