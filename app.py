import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================================================
# Page Configuration (MUST be first)
# =========================================================
st.set_page_config(
    page_title="Employee Retention Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Custom CSS – Dark Corporate Theme (STABLE)
# =========================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #FFFFFF;
    }

    .stSidebar {
        background-color: #111827;
    }

    .stButton > button {
        background-color: #2C7BE5;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1A68D1;
    }

    .result-box {
        background-color: #111827;
        padding: 20px;
        border-radius: 14px;
        border-left: 6px solid #2C7BE5;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
st.sidebar.title("Employee Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
relevent_experience = st.sidebar.selectbox(
    "Relevant Experience",
    ["Has relevent experience", "No relevent experience"]
)
education_level = st.sidebar.selectbox(
    "Education Level",
    ["Graduate", "Masters", "High School", "Phd"]
)
enrolled_university = st.sidebar.selectbox(
    "University Enrollment",
    ["no_enrollment", "Full time course", "Part time course"]
)
company_type = st.sidebar.selectbox(
    "Company Type",
    ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Other"]
)

company_size_grouped = st.sidebar.selectbox(
    "Company Size",
    ["Small", "Medium", "Large", "Unknown"]
)

city_development_index = st.sidebar.slider(
    "City Development Index",
    0.0, 1.0, 0.6, 0.01
)

experience = st.sidebar.number_input(
    "Total Experience (Years)",
    min_value=0, max_value=30, value=5
)

training_hours = st.sidebar.number_input(
    "Training Hours Completed",
    min_value=0, max_value=400, value=120, step=5
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
    "This application estimates the **probability of an employee seeking a job change** "
    "based on demographic, educational, and professional attributes."
)

# =========================================================
# Prediction Logic
# =========================================================
if st.button("Predict Job Change Probability"):

    input_data = {
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "company_type": company_type,
        "company_size_grouped": company_size_grouped,
        "city_development_index": city_development_index,
        "experience": experience,
        "training_hours": training_hours,
        "lastnewjob": lastnewjob
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align with training columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    # =====================================================
    # Result Display
    # =====================================================
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.metric("Job Change Probability", f"{probability:.2%}")

    if probability < 0.30:
        st.success("Low risk of job change")
    elif probability < 0.60:
        st.warning("Medium risk of job change")
    else:
        st.error("High risk of job change")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # Visualization: Probability Gauge (Clean)
    # =====================================================
    st.subheader("Risk Visualization")

    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.barh(["Job Change Risk"], [probability])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability Score")
    ax.set_title("Predicted Job Change Probability")
    st.pyplot(fig)

# =========================================================
# Explanation Section
# =========================================================
with st.expander("Why probability-based prediction?"):
    st.info(
        "Employee retention is a risk assessment problem. "
        "Probability scores allow HR teams to prioritize employees for engagement "
        "and retention strategies instead of relying on a binary decision."
    )

# =========================================================
# Footer
# =========================================================
st.caption(
    "This tool is intended for decision support only. "
    "Predictions are based on historical employee behavior patterns."
)
