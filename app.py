# ==============================
# IMPORTS
# ==============================
import streamlit as st
import joblib
import tempfile
import base64
import os
import matplotlib.pyplot as plt
import numpy as np

from analysis.compute_metrics import compute_metrics
from analysis.subgroup_risk import compute_subgroup_risk
from analysis.bias_severity import compute_bias_severity


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="ModelGuard AI", layout="centered")


# ==============================
# LOAD BACKGROUND IMAGE
# ==============================
def get_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

bg_image = get_base64("assets/bg.png")


# ==============================
# CUSTOM CSS
# ==============================
if bg_image:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.85);
        z-index: -1;
    }}

    section[data-testid="stSidebar"] {{display: none;}}
    [data-testid="collapsedControl"] {{display: none;}}

    h1, h2, h3 {{
        color: #00f7ff;
        text-align: center;
        font-family: 'Courier New', monospace;
    }}

    div.stButton > button {{
        background-color: #00f7ff;
        color: black;
        font-weight: bold;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }}

    div.stButton > button:hover {{
        background-color: #00c3cc;
        transform: scale(1.05);
    }}

    </style>
    """, unsafe_allow_html=True)


# ==============================
# SESSION STATE
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "upload"

if "show_graph" not in st.session_state:
    st.session_state.show_graph = False


# =========================================================
# UPLOAD PAGE
# =========================================================
if st.session_state.page == "upload":

    st.title("ModelGuard AI Risk Engine")
    st.markdown("### 🤖 Automated Deployment Intelligence System")
    st.markdown("---")

    old_model = st.file_uploader("📂 Upload Baseline Model (.pkl)", type=["pkl"])
    new_model = st.file_uploader("📂 Upload Candidate Model (.pkl)", type=["pkl"])

    if st.button("⚡ Run Deployment Risk Analysis"):

        if old_model and new_model:

            temp_old = tempfile.NamedTemporaryFile(delete=False)
            temp_old.write(old_model.read())
            temp_old.close()

            temp_new = tempfile.NamedTemporaryFile(delete=False)
            temp_new.write(new_model.read())
            temp_new.close()

            st.session_state.baseline_path = temp_old.name
            st.session_state.updated_path = temp_new.name
            st.session_state.page = "results"
            st.rerun()

        else:
            st.warning("Please upload both models.")


# =========================================================
# RESULTS PAGE
# =========================================================
elif st.session_state.page == "results":

    st.title("📊 AI Risk Intelligence Report")

    try:
        with st.spinner("🔄 Running AI Risk Diagnostics..."):

            baseline_model = joblib.load(st.session_state.baseline_path)
            updated_model = joblib.load(st.session_state.updated_path)

            X_test = joblib.load("data/X_test.pkl")
            y_test = joblib.load("data/y_test.pkl")

            flip_rate, conf_shift, feature_drift = compute_metrics(
                baseline_model,
                updated_model,
                X_test
            )

            subgroup_risk = compute_subgroup_risk(
                baseline_model,
                updated_model,
                X_test,
                y_test,
                feature="mean radius"
            )

            bias_severity = compute_bias_severity(
                updated_model,
                X_test,
                y_test,
                feature="mean radius"
            )

            # FINAL RISK SCORE (0–1)
            final_risk_score = (
                0.30 * flip_rate +
                0.25 * conf_shift +
                0.20 * feature_drift +
                0.15 * subgroup_risk +
                0.10 * bias_severity
            )

        st.markdown("---")
        st.subheader("🧠 Risk Components")

        col1, col2 = st.columns(2)

        col1.metric("Prediction Flip Rate", round(flip_rate, 4))
        col1.metric("Confidence Shift", round(conf_shift, 4))
        col1.metric("Feature Drift", round(feature_drift, 4))

        col2.metric("Subgroup Risk", round(subgroup_risk, 4))
        col2.metric("Bias Severity Score", round(bias_severity, 4))

        st.markdown("---")
        st.subheader("🎯 Final Risk Score (0 – 1 Scale)")
        st.metric("Overall Risk Score", round(final_risk_score, 4))
        st.progress(min(final_risk_score, 1.0))

        st.markdown("---")
        st.subheader("🚦 Deployment Recommendation")

        # Realistic thresholds
        if final_risk_score < 0.02:
            st.success("✅ SAFE TO DEPLOY (Automatic Deployment Triggered)")
            deployment_status = "DEPLOY"
        elif final_risk_score < 0.07:
            st.warning("⚠ MEDIUM RISK – Manual Approval Required")
            deployment_status = "REVIEW"
        else:
            st.error("❌ HIGH RISK – Automatic ROLLBACK Triggered")
            deployment_status = "ROLLBACK"

        st.markdown("---")

        # =============================
        # GRAPH TOGGLE BUTTON
        # =============================
        if st.button("📊  Risk Graph"):
            st.session_state.show_graph = not st.session_state.show_graph

        if st.session_state.show_graph:

            components = [
                flip_rate,
                conf_shift,
                feature_drift,
                subgroup_risk,
                bias_severity
            ]

            labels = [
                "Flip",
                "Confidence",
                "Drift",
                "Subgroup",
                "Bias"
            ]

            fig, ax = plt.subplots(figsize=(8, 5))

            bars = ax.bar(labels, components)

            ax.set_ylim(0, max(components) + 0.02)
            ax.set_title("Risk Component Distribution", fontsize=14)
            ax.set_ylabel("Risk Score")

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f"{height:.3f}",
                    ha='center',
                    va='bottom'
                )

            ax.grid(axis='y', linestyle='--', alpha=0.4)

            st.pyplot(fig)

        st.markdown("---")

        if st.button("⬅ Back to Model Upload"):
            st.session_state.page = "upload"
            st.rerun()

    except Exception as e:
        st.error("❌ Error loading models or computing risk.")
        st.error(str(e))