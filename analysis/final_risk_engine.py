import joblib
from analysis.compute_metrics import compute_metrics
from analysis.subgroup_risk import compute_subgroup_risk


# ===============================
# Risk Score Computation Function
# ===============================
def compute_final_risk_score(
    prediction_flip_rate,
    confidence_shift,
    feature_drift,
    subgroup_risk
):
    """
    Computes the final deployment risk score
    using weighted aggregation of risk elements.
    """

    # Weights (configurable policy)
    w_flip = 0.40
    w_conf = 0.30
    w_feat = 0.20
    w_sub = 0.10

    risk_score = (
        w_flip * prediction_flip_rate +
        w_conf * confidence_shift +
        w_feat * feature_drift +
        w_sub * subgroup_risk
    )

    return risk_score


# ===============================
# Explainability Layer
# ===============================
def explain_risk(
    prediction_flip_rate,
    confidence_shift,
    feature_drift,
    subgroup_risk
):
    """
    Generates human-readable explanations
    for the deployment risk decision.
    """

    explanations = []

    if prediction_flip_rate > 0.15:
        explanations.append(
            "High prediction instability detected between model versions"
        )

    if confidence_shift > 0.10:
        explanations.append(
            "Significant confidence shift observed in model predictions"
        )

    if feature_drift > 0.05:
        explanations.append(
            "Model reasoning changed significantly (feature importance drift)"
        )

    if subgroup_risk > 0.05:
        explanations.append(
            "Performance degradation detected in specific data subgroups"
        )

    if not explanations:
        explanations.append(
            "No significant behavioral risks detected in model update"
        )

    return explanations


# ===============================
# MAIN EXECUTION
# ===============================

# Load test data
X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

# Load baseline model
model_v1 = joblib.load("models/model_v1.pkl")

# 🔁 Select updated model version here
model_v2 = joblib.load("models/model_v2_safe.pkl")
# model_v2 = joblib.load("models/model_v2_caution.pkl")
# model_v2 = joblib.load("models/model_v2_block.pkl")

# Compute core risk metrics
prediction_flip_rate, confidence_shift, feature_drift = compute_metrics(
    model_v1, model_v2, X_test
)

subgroup_risk = compute_subgroup_risk(
    model_v1,
    model_v2,
    X_test,
    y_test,
    feature="mean radius"
)

# Compute final risk score
final_risk_score = compute_final_risk_score(
    prediction_flip_rate,
    confidence_shift,
    feature_drift,
    subgroup_risk
)

# Deployment decision
if final_risk_score < 0.30:
    decision = "✅ Safe to Deploy"
elif final_risk_score < 0.60:
    decision = "⚠️ Deploy with Caution"
else:
    decision = "❌ Block Deployment"

# Explainability
explanations = explain_risk(
    prediction_flip_rate,
    confidence_shift,
    feature_drift,
    subgroup_risk
)

# ===============================
# OUTPUT
# ===============================
print("\n====== MODEL UPDATE RISK ASSESSMENT ======")
print(f"Prediction Flip Rate : {prediction_flip_rate:.4f}")
print(f"Confidence Shift     : {confidence_shift:.4f}")
print(f"Feature Drift        : {feature_drift:.6f}")
print(f"Subgroup Risk        : {subgroup_risk:.4f}")

print("\nFinal Risk Score     :", round(final_risk_score, 4))
print("Deployment Decision :", decision)

print("\nRisk Explanation:")
for exp in explanations:
    print("-", exp)
