import streamlit as st
import joblib

from analysis.compute_metrics import compute_metrics
from analysis.subgroup_risk import compute_subgroup_risk
from analysis.bias_severity import compute_bias_severity

st.set_page_config(page_title="Risk Results", layout="centered")

st.title("Model Risk Evaluation Results")

# Check if models exist in session
if "baseline_path" not in st.session_state or "updated_path" not in st.session_state:
    st.error("No models found. Please go back and upload models.")
    st.stop()

# Load models
baseline_model = joblib.load(st.session_state["baseline_path"])
updated_model = joblib.load(st.session_state["updated_path"])

# Load dataset
X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

# =========================
# Compute Metrics
# =========================
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

final_risk_score = (
    0.30 * flip_rate +
    0.25 * conf_shift +
    0.20 * feature_drift +
    0.15 * subgroup_risk +
    0.10 * bias_severity
)

# =========================
# Display Results
# =========================
st.subheader("Risk Components")

st.metric("Prediction Flip Rate", round(flip_rate, 3))
st.metric("Confidence Shift", round(conf_shift, 3))
st.metric("Feature Drift", round(feature_drift, 6))
st.metric("Subgroup Risk", round(subgroup_risk, 3))
st.metric("Bias Severity Score", round(bias_severity, 3))

st.divider()

st.subheader("Final Risk Score")
st.metric("Risk Score", round(final_risk_score, 4))
st.progress(min(final_risk_score, 1.0))

st.subheader("Deployment Recommendation")

if final_risk_score < 0.30:
    st.success("✅ Safe to Deploy")
elif final_risk_score < 0.60:
    st.warning("⚠️ Deploy with Caution")
else:
    st.error("❌ Not Safe to Deploy")

st.button("⬅ Back", on_click=lambda: st.switch_page("app.py"))