import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💼",
    layout="centered"
)

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    model    = joblib.load("salary_prediction_model.pkl")
    encoders = joblib.load("salary_label_encoders.pkl")
    scaler   = joblib.load("salary_scaler.pkl")
    return model, encoders, scaler

model, label_encoders, scaler = load_artifacts()

# ── TITLE ──────────────────────────────────────────────────────────────────────

st.title("💼 Salary Prediction System")
st.write("Fill in the employee details below and click **Predict Salary**.")
st.divider()

# ── INPUT FORM ─────────────────────────────────────────────────────────────────

with st.form("salary_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Info")

        age = st.number_input("Age", min_value=18, max_value=65, value=28)

        gender = st.selectbox("Gender", ["Male", "Female"])

        education = st.selectbox(
            "Education Level",
            ["High School", "Bachelor's", "Master's", "PhD"]
        )

        department = st.selectbox(
            "Department",
            ["IT", "Finance", "HR", "Marketing", "Operations"]
        )

    with col2:
        st.subheader("💼 Work Details")

        job_role = st.selectbox(
            "Job Role",
            ["Intern", "Analyst", "Engineer", "Manager", "Director"]
        )

        experience = st.slider("Years of Experience", 0, 30, 5)

        job_level = st.selectbox(
            "Job Level",
            ["Entry", "Mid", "Senior"]
        )

        remote_work = st.selectbox("Remote Work", ["Yes", "No"])

        certifications = st.number_input(
            "Number of Certifications", min_value=0, max_value=10, value=1
        )

        performance_score = st.slider("Performance Score (1–10)", 1, 10, 7)

    st.divider()
    submitted = st.form_submit_button("🔍 Predict Salary", use_container_width=True)

# ── PREDICTION ─────────────────────────────────────────────────────────────────

if submitted:

    # LabelEncoder maps alphabetically → replicate exactly
    gender_map  = {"Female": 0, "Male": 1}
    edu_map     = {"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3}
    dept_map    = {"Finance": 0, "HR": 1, "IT": 2, "Marketing": 3, "Operations": 4}
    role_map    = {"Analyst": 0, "Director": 1, "Engineer": 2, "Intern": 3, "Manager": 4}
    level_map   = {"Entry": 0, "Mid": 1, "Senior": 2}
    yes_no_map  = {"No": 0, "Yes": 1}

    input_data = pd.DataFrame([{
        "Age":               age,
        "Gender":            gender_map[gender],
        "Education":         edu_map[education],
        "Job_Role":          role_map[job_role],
        "Department":        dept_map[department],
        "Experience":        experience,
        "Job_Level":         level_map[job_level],
        "Remote_Work":       yes_no_map[remote_work],
        "Certifications":    certifications,
        "Performance_Score": performance_score,
    }])

    # Predict
    predicted_salary = model.predict(input_data)[0]
    predicted_salary = max(0, predicted_salary)

    # ── Salary band label ──
    if predicted_salary < 40_000:
        band       = "🟡 Entry-Level Band"
        band_color = "warning"
    elif predicted_salary < 80_000:
        band       = "🔵 Mid-Level Band"
        band_color = "info"
    elif predicted_salary < 130_000:
        band       = "🟢 Senior-Level Band"
        band_color = "success"
    else:
        band       = "🏆 Executive Band"
        band_color = "success"

    # ── Display result ──
    st.divider()
    st.subheader("📊 Prediction Result")

    if band_color == "success":
        st.success(f"{band}  —  Predicted Annual Salary: **${predicted_salary:,.0f}**")
    elif band_color == "info":
        st.info(f"{band}  —  Predicted Annual Salary: **${predicted_salary:,.0f}**")
    else:
        st.warning(f"{band}  —  Predicted Annual Salary: **${predicted_salary:,.0f}**")

    # ── Metric cards ──
    monthly = predicted_salary / 12
    weekly  = predicted_salary / 52
    daily   = predicted_salary / 260

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual",  f"${predicted_salary:,.0f}")
    c2.metric("Monthly", f"${monthly:,.0f}")
    c3.metric("Weekly",  f"${weekly:,.0f}")
    c4.metric("Daily",   f"${daily:,.0f}")

    # ── Salary progress bar (scaled to $200k max) ──
    st.write("**Salary Scale** (up to $200,000)")
    st.progress(min(int((predicted_salary / 200_000) * 100), 100))

    # ── Key factors summary ──
    st.divider()
    st.subheader("🔍 Key Factors Summary")

    factors_df = pd.DataFrame({
        "Factor":      ["Education", "Job Role", "Experience (yrs)", "Job Level", "Performance Score", "Certifications"],
        "Your Input":  [education, job_role, experience, job_level, performance_score, certifications],
        "Impact":      ["High ⬆", "High ⬆", "High ⬆", "Medium ⬆", "Medium ⬆", "Medium ⬆"]
    })
    st.table(factors_df)

    with st.expander("🔎 View encoded inputs sent to model"):
        st.dataframe(input_data)
