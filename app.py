import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Employee Retention Prediction",
    layout="wide"
)

# =========================================================
# Custom CSS – Dark Blue Corporate Theme
# =========================================================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: #FFFFFF;
}

h1, h2, h3 {
    color: #FFFFFF;
}

.stSidebar {
    background-color: #111827;
}

.stButton>button {
    background-color: #2C7BE5;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
}

.stButton>button:hover {
    background-color: #1A68D1;
}

.metric-box {
    background-color: #111827;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #2C7BE5;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Load Model & Feature Columns
# =========================================================
with open("lgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# =========================================================
# Sidebar – Employee Inputs
# =========================================================
st.sidebar.header("Employee Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
relevent_experience = st.sidebar.selectbox(
    "Relevant Experience",
    ["Has relevent experience", "No relevent experience"]
)
enrolled_university = st.sidebar.selectbox(
    "University Enrollment",
    ["no_enrollment", "Part time course", "Full time course"]
)
education_level = st.sidebar.selectbox(
    "Education Level",
    ["Graduate", "Masters", "High School", "Phd"]
)
company_type = st.sidebar.selectbox(
    "Company Type",
    ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Other"]
)
city_development_index = st.sidebar.slider(
    "City Development Index",
    0.0, 1.0, 0.5, 0.01
)
experience = st.sidebar.number_input(
    "Total Experience (Years)", 0, 30, 5
)
training_hours = st.sidebar.number_input(
    "Training Hours Completed", 0, 400, 120, step=5
)
lastnewjob = st.sidebar.selectbox(
    "Years Since Last Job Change",
    ["Never", "1", "2", "3", "4", ">4"]
)

# =========================================================
# Main Section
# =========================================================
st.title("Employee Retention Prediction")

st.write(
    "This application estimates the probability that an employee is likely "
    "to look for a job change based on historical workforce data."
)

# =========================================================
# Prediction
# =========================================================
if st.button("Predict Job Change Probability"):

    input_data = {
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "company_type": company_type,
        "city_development_index": city_development_index,
        "experience": experience,
        "training_hours": training_hours,
        "lastnewjob": lastnewjob
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align with training features
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    # =====================================================
    # Result Display
    # =====================================================
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.metric("Job Change Probability", f"{probability:.2%}")

    if probability < 0.30:
        st.success("Low risk of job change")
    elif probability < 0.60:
        st.warning("Medium risk of job change")
    else:
        st.error("High risk of job change")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # Visualization 1: Probability Bar
    # =====================================================
    st.subheader("Risk Visualization")

    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.barh(["Job Change Risk"], [probability])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Predicted Job Change Probability")
    st.pyplot(fig)

# =========================================================
# Optional: Dataset Insights (Static / Reference)
# =========================================================
with st.expander("Why probability-based prediction?"):
    st.info(
        "Employee retention is a risk-based problem. "
        "Probability scores help HR teams prioritize employees who may require "
        "engagement or retention strategies, rather than relying on a simple yes/no prediction."
    )

# =========================================================
# Footer
# =========================================================
st.caption(
    "This tool is intended for decision support only. "
    "Predictions are based on historical patterns and may not reflect external market conditions."
)
