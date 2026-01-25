import streamlit as st
import joblib

from analysis.compute_metrics import compute_metrics
from analysis.subgroup_risk import compute_subgroup_risk
from analysis.bias_severity import compute_bias_severity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="ModelGuard", layout="centered")

st.title("ModelGuard – Deployment Risk Assessment")

# =========================
# LOAD DATA
# =========================
X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

# =========================
# LOAD BASELINE MODEL
# =========================
baseline_model = joblib.load("models/model_v0.pkl")

# =========================
# MODEL VERSION SELECTION
# =========================
model_version = st.selectbox(
    "Select Updated Model Version",
    ["v1", "v2", "v3"]
)

model_paths = {
    "v1": "models/model_v1.pkl",   # no bias
    "v2": "models/model_v2.pkl",   # moderate bias
    "v3": "models/model_v3.pkl"    # severe bias
}

updated_model = joblib.load(model_paths[model_version])

# =========================
# COMPUTE CORE METRICS
# =========================
flip_rate, conf_shift, feature_drift = compute_metrics(
    baseline_model, updated_model, X_test
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

# =========================
# FINAL RISK SCORE (UPDATED)
# =========================
final_risk_score = (
    0.30 * flip_rate +
    0.25 * conf_shift +
    0.20 * feature_drift +
    0.15 * subgroup_risk +
    0.10 * bias_severity
)

# =========================
# DISPLAY METRICS
# =========================
st.subheader("Risk Components")

st.metric("Prediction Flip Rate", round(flip_rate, 3))
st.metric("Confidence Shift", round(conf_shift, 3))
st.metric("Feature Drift", round(feature_drift, 6))
st.metric("Subgroup Risk", round(subgroup_risk, 3))
st.metric("Bias Severity Score", round(bias_severity, 3))

st.divider()

# =========================
# DISPLAY FINAL RISK SCORE
# =========================
st.subheader("Final Risk Score")
st.metric("Risk Score", round(final_risk_score, 4))
st.progress(min(final_risk_score, 1.0))

# =========================
# DEPLOYMENT DECISION
# =========================
st.subheader("Deployment Decision")

if final_risk_score < 0.030:
    st.success("✅ Safe to Deploy")
elif final_risk_score < 0.080:
    st.warning("⚠️ Deploy with Caution")
else:
    st.error("❌ Not Safe to Deploy")

# =========================
# EXPLANATION (IMPORTANT)
# =========================
st.subheader("Risk Explanation")

if subgroup_risk > 0.10:
    st.write("• Performance degradation detected in specific subgroups")

if bias_severity > 0.15:
    st.write("• Severe bias detected due to training data imbalance")

if final_risk_score < 0.30:
    st.write("• Model behavior remains stable across updates")

