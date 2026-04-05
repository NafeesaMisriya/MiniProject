import base64
import os
import tempfile

import joblib
import matplotlib.pyplot as plt
import streamlit as st

from analysis.compute_metrics import compute_metrics
from analysis.subgroup_risk import compute_subgroup_risk
from analysis.bias_severity import compute_bias_severity
from backend.database import get_connection


CURRENT_MODEL_PATH = "models/current.pkl"


def read_current_model():
    try:
        with open(CURRENT_MODEL_PATH, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "None set"


def write_current_model(model_name: str):
    os.makedirs("models", exist_ok=True)
    with open(CURRENT_MODEL_PATH, "w") as f:
        f.write(f"Active Model: {model_name}")


# ✅ NEW FUNCTION (RESET)
def reset_current_model():
    os.makedirs("models", exist_ok=True)
    with open(CURRENT_MODEL_PATH, "w") as f:
        f.write("Active Model: None")


st.set_page_config(page_title="Model Risk Assessment", layout="centered")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    _img = _b64("assets/bg.png")
    _bg  = (
        f"background-image:url('data:image/png;base64,{_img}');"
        "background-size:cover;background-position:center;background-attachment:fixed;"
    )
except FileNotFoundError:
    _bg = "background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);"


def _nav_row(label, page_key):
    active = st.session_state.page == page_key
    if active:
        st.sidebar.markdown(f"""
            <div style="
                background:rgba(255,255,255,0.16);
                border-left:3px solid rgba(255,255,255,0.85);
                color:white;
                font-weight:600;
                font-size:15px;
                padding:10px 16px;
                border-radius:4px;
                margin:1px 0;">
                {label}
            </div>""",
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(label, key=f"nav_{page_key}"):
            st.session_state.page = page_key
            st.rerun()


st.markdown(f"""
<style>
div[data-testid="stSidebarNav"] {{ display: none !important; }}
.stApp, .stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section, [data-testid="stMain"] {{
    {_bg}
    color: white;
}}
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] > div {{
    background: transparent !important;
}}
div[data-testid="stSidebar"] {{
    background-color: rgba(15,32,39,0.90) !important;
}}
</style>
""", unsafe_allow_html=True)


# ✅ SESSION STATE (added baseline_set)
for _k, _v in [
    ("page", "app"),
    ("history", []),
    ("show_graph", False),
    ("candidate_name", None),
    ("baseline_name", None),
    ("review_acted", False),
    ("baseline_set", False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ================= SIDEBAR =================
st.sidebar.title("Dashboard")

_active = read_current_model()
st.sidebar.markdown(
    f"<div style='color:rgba(255,255,255,0.55);font-size:12px;padding:4px 16px;'>🟢 {_active}</div>",
    unsafe_allow_html=True
)

_nav_row("📤 Upload", "app")
_nav_row("📊 Results", "results")
_nav_row("🕓 Deployment History", "history")


# ================= PAGE 1 =================
if st.session_state.page == "app":

    # ✅ Reset when entering page
    if not st.session_state.baseline_set:
        reset_current_model()

    st.title("ModelGuard AI Risk Engine")

    baseline_file  = st.file_uploader("Upload Baseline Model (.pkl)",  type=["pkl"])

    # ✅ Immediate baseline activation
    if baseline_file is not None:
        baseline_name = os.path.splitext(baseline_file.name)[0]

        if (not st.session_state.baseline_set or
            st.session_state.baseline_name != baseline_name):

            write_current_model(baseline_name)
            st.session_state.baseline_name = baseline_name
            st.session_state.baseline_set = True
            st.rerun()

    candidate_file = st.file_uploader("Upload Candidate Model (.pkl)", type=["pkl"])

    if st.button("Run Risk Analysis"):

        if baseline_file is None or candidate_file is None:
            st.warning("Please upload both models.")
            st.stop()

        baseline_name  = os.path.splitext(baseline_file.name)[0]
        candidate_name = os.path.splitext(candidate_file.name)[0]

        st.session_state.baseline_name  = baseline_name
        st.session_state.candidate_name = candidate_name
        st.session_state.review_acted   = False

        tmp_old = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp_old.write(baseline_file.read()); tmp_old.close()

        tmp_new = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp_new.write(candidate_file.read()); tmp_new.close()

        baseline_model = joblib.load(tmp_old.name)
        updated_model  = joblib.load(tmp_new.name)

        X_test = joblib.load("data/X_test.pkl")
        y_test = joblib.load("data/y_test.pkl")

        flip_rate, conf_shift, feature_drift = compute_metrics(
            baseline_model, updated_model, X_test
        )
        subgroup_risk = compute_subgroup_risk(
            baseline_model, updated_model, X_test, y_test, feature="mean radius"
        )
        bias_severity = compute_bias_severity(
            updated_model, X_test, y_test, feature="mean radius"
        )

        final_risk = (
            0.30 * flip_rate +
            0.25 * conf_shift +
            0.20 * feature_drift +
            0.15 * subgroup_risk +
            0.10 * bias_severity
        )

        decision = (
            "DEPLOY" if final_risk < 0.02 else
            "REVIEW" if final_risk < 0.07 else
            "ROLLBACK"
        )

        # existing logic kept
        if decision == "DEPLOY":
            write_current_model(candidate_name)
        elif decision == "ROLLBACK":
            write_current_model(baseline_name)

        result = dict(
            flip_rate=flip_rate,
            conf_shift=conf_shift,
            feature_drift=feature_drift,
            subgroup_risk=subgroup_risk,
            bias_severity=bias_severity,
            final_risk=final_risk,
            decision=decision,
            candidate_name=candidate_name,
            baseline_name=baseline_name,
        )

        st.session_state.results = result
        st.session_state.show_graph = False
        st.session_state.history.append(result)

        # DB unchanged
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO comparisons
               (baseline_model, updated_model, flip_rate, confidence_shift,
                feature_drift, subgroup_risk, bias_severity, final_risk, decision)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                baseline_name, candidate_name,
                flip_rate, conf_shift,
                feature_drift, subgroup_risk,
                bias_severity, final_risk, decision
            )
        )
        conn.commit()
        conn.close()

        st.session_state.page = "results"
        st.rerun()


# ================= PAGE 2 =================
elif st.session_state.page == "results":

    st.title("AI Risk Intelligence Report")

    if "results" not in st.session_state:
        st.warning("No results yet.")
        st.stop()

    r = st.session_state.results
    cand = r["candidate_name"]
    base = r["baseline_name"]
    decision = r["decision"]

    st.subheader("Risk Metrics")
    c1, c2 = st.columns(2)

    c1.metric("Flip Rate", round(r["flip_rate"], 4))
    c1.metric("Confidence Shift", round(r["conf_shift"], 4))
    c1.metric("Feature Drift", round(r["feature_drift"], 4))

    c2.metric("Subgroup Risk", round(r["subgroup_risk"], 4))
    c2.metric("Bias Severity", round(r["bias_severity"], 4))

    st.subheader("Final Risk Score")
    st.metric("Overall Risk", round(r["final_risk"], 4))

    st.markdown("---")
    st.subheader("Deployment Decision")

    if decision == "DEPLOY":
        st.success("✅ SAFE TO DEPLOY")
        st.info(f"Active Model: {read_current_model()}")

    elif decision == "REVIEW":
        st.warning("⚠️ DEPLOY WITH CAUTION")

        if not st.session_state.review_acted:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Accept Candidate"):
                    write_current_model(cand)
                    st.session_state.review_acted = True
                    st.rerun()

            with col2:
                if st.button("❌ Rollback to Baseline"):
                    write_current_model(base)
                    st.session_state.review_acted = True
                    st.rerun()

        st.info(f"Active Model: {read_current_model()}")

    else:
        st.error("🚨 ROLLBACK TRIGGERED")
        st.info(f"Active Model: {read_current_model()}")

    st.markdown("---")

    tog = "Hide Graph" if st.session_state.show_graph else "Show Graph"
    if st.button(tog):
        st.session_state.show_graph = not st.session_state.show_graph
        st.rerun()

    if st.session_state.show_graph:
        labels = ["Flip", "Conf", "Drift", "Subgroup", "Bias"]
        values = [
            r["flip_rate"], r["conf_shift"],
            r["feature_drift"], r["subgroup_risk"],
            r["bias_severity"]
        ]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, values)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, val, f"{val:.4f}", ha='center')
        st.pyplot(fig)

    st.markdown("---")

    # ✅ Reset on back
    if st.button("⬅ Back"):
        reset_current_model()
        st.session_state.baseline_set = False
        st.session_state.page = "app"
        st.rerun()


# ================= PAGE 3 =================
elif st.session_state.page == "history":

    st.title("🕓 Deployment History")

    if not st.session_state.history:
        st.info("No runs yet.")
        st.stop()

    for i, h in enumerate(st.session_state.history):
        with st.expander(
            f"Run {i+1} | {h['candidate_name']} | Risk {round(h['final_risk'],4)} | {h['decision']}"
        ):
            st.write(h)

    if st.button("⬅ Back"):
        reset_current_model()
        st.session_state.baseline_set = False
        st.session_state.page = "app"
        st.rerun()