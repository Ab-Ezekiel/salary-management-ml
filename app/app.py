import streamlit as st
import pandas as pd
from model import predict

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Payroll ML System", layout="wide")

# ================================
# SESSION STATE (IMPORTANT)
# ================================
# Session state initialization
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("payroll_dataset_fixed.csv")

if "detection_run" not in st.session_state:
    st.session_state.detection_run = False
# ================================
# SIDEBAR
# ================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Anomaly Detection", "Fairness Analysis"]
)

# ================================
# OVERVIEW PAGE
# ================================
if page == "Overview":

    df = st.session_state.df
    st.title("📊 Payroll Overview")

    # Filters
    with st.sidebar:
        st.subheader("Filters")

        gender_filter = st.selectbox(
            "Select Gender",
            ["All"] + list(df["gender"].unique())
        )

        dept_filter = st.selectbox(
            "Select Department",
            ["All"] + list(df["department"].unique())
        )

    filtered_df = df.copy()

    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["gender"] == gender_filter]

    if dept_filter != "All":
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]

    st.dataframe(filtered_df)

# ================================
# ANOMALY DETECTION PAGE
# ================================
elif page == "Anomaly Detection":

    df = st.session_state.df
    st.title("🚨 Anomaly Detection")

    # Buttons (side by side)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Detection"):
            data = st.session_state.df.copy()

            preds, probs = predict(data)

            st.session_state.df["anomaly_prediction"] = preds
            st.session_state.df["anomaly_score"] = probs
            st.session_state.detection_run = True

            st.success("Detection completed!")

    with col2:
        if st.button("Reset Detection"):
            st.session_state.df = pd.read_csv("payroll_dataset_fixed.csv")
            st.session_state.detection_run = False
            st.warning("Detection reset!")

    # ================================
    # DISPLAY RESULTS (IMPORTANT)
    # ================================
    if st.session_state.detection_run:

        df = st.session_state.df

        # Summary
        st.subheader("📊 Summary")
        st.bar_chart(df["anomaly_prediction"].value_counts())

        st.subheader("Average Anomaly Score")
        st.write(df["anomaly_score"].mean())

        # Flagged employees
        st.subheader("🚨 Flagged Employees")
        flagged = df[df["anomaly_prediction"] == 1]
        st.dataframe(flagged)

    else:
        st.info("Click 'Run Detection' to start analysis.")

    # ================================
    # SEARCH
    # ================================
    st.subheader("🔍 Search Employee")

    emp_id = st.text_input("Enter Employee ID")

    if emp_id:
        try:
            result = df[df["employee_id"] == int(emp_id)]
            st.dataframe(result)
        except:
            st.warning("Invalid Employee ID")

# ================================
# FAIRNESS PAGE
# ================================
elif page == "Fairness Analysis":

    df = st.session_state.df
    st.title("⚖️ Fairness Analysis")

    if not st.session_state.detection_run:
        st.warning("Run anomaly detection first!")
    else:

        st.info("Selection rate shows the proportion of employees flagged as anomalies within each group. Significant disparities may indicate bias in the model's predictions.")

        # ======================
        # Gender Fairness
        # ======================
        st.subheader("Gender Fairness")

        gender_fairness = df.groupby("gender")["anomaly_prediction"].mean()

        st.bar_chart(gender_fairness)

        st.write(
            gender_fairness
            .round(3)
            .to_frame()
            .rename(columns={"anomaly_prediction": "Selection Rate"})
        )

        # ======================
        # Department Fairness
        # ======================
        st.subheader("Department Fairness")

        dept_fairness = df.groupby("department")["anomaly_prediction"].mean()

        st.bar_chart(dept_fairness)

        st.write(
            dept_fairness
            .round(3)
            .to_frame()
            .rename(columns={"anomaly_prediction": "Selection Rate"})
        )

    