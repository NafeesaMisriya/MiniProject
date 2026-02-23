# ==============================
# IMPORTS (MUST BE AT TOP)
# ==============================
import streamlit as st
import joblib
import tempfile
import base64

from analysis.compute_metrics import compute_metrics
from analysis.subgroup_risk import compute_subgroup_risk
from analysis.bias_severity import compute_bias_severity


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="ModelGuard AI", layout="centered")


# ==============================
# LOAD PNG BACKGROUND
# ==============================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("assets/bg.png")


# ==============================
# CUSTOM CSS
# ==============================
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
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.75);
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

.stFileUploader {{
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
}}

</style>
""", unsafe_allow_html=True)


# ==============================
# PAGE STATE CONTROL
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "upload"


# =========================================================
# UPLOAD PAGE
# =========================================================
if st.session_state.page == "upload":

    st.title(" ModelGuard AI Risk Engine")
    st.markdown("### 🤖 AI Deployment Intelligence System")

    st.markdown("---")

    old_model = st.file_uploader("📂 Upload Baseline Model (.pkl)", type=["pkl"])
    new_model = st.file_uploader("📂 Upload Candidate Model (.pkl)", type=["pkl"])

    if st.button("⚡ Run Deployment Risk Analysis"):

        if old_model and new_model:

            temp_old = tempfile.NamedTemporaryFile(delete=False)
            temp_old.write(old_model.read())

            temp_new = tempfile.NamedTemporaryFile(delete=False)
            temp_new.write(new_model.read())

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

        final_risk_score = (
            0.30 * flip_rate +
            0.25 * conf_shift +
            0.20 * feature_drift +
            0.15 * subgroup_risk +
            0.10 * bias_severity
        )

    st.markdown("---")
    st.subheader("🧠 Final Risk Score")

    st.metric("Overall Risk Score", round(final_risk_score, 4))
    st.progress(min(final_risk_score, 1.0))

    st.markdown("---")
    st.subheader("🚦 Deployment Recommendation")

    if final_risk_score < 0.30:
        st.success("✅ SAFE TO DEPLOY")
    elif final_risk_score < 0.60:
        st.warning("⚠ DEPLOY WITH CAUTION")
    else:
        st.error("❌ NOT SAFE FOR PRODUCTION")

    if st.button("⬅ Back to Model Upload"):
        st.session_state.page = "upload"
        st.rerun()