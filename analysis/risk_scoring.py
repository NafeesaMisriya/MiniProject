import numpy as np

# ===== INPUT METRICS =====
prediction_flip_rate = 0.18
confidence_shift = 0.12
feature_drift_score = 0.000083

# ===== WEIGHTS =====
w_flip = 0.5
w_conf = 0.3
w_feat = 0.2

# ===== RISK SCORE =====
risk_score = (
    w_flip * prediction_flip_rate +
    w_conf * confidence_shift +
    w_feat * feature_drift_score
)

print("Overall Risk Score:", round(risk_score, 4))

# ===== DEPLOYMENT DECISION =====
if risk_score < 0.30:
    decision = "✅ Safe to Deploy"
elif risk_score < 0.60:
    decision = "⚠️ Deploy with Caution"
else:
    decision = "❌ Block Deployment"

print("Deployment Decision:", decision)

# ===== EXPLANATION =====
reasons = []

if prediction_flip_rate > 0.15:
    reasons.append("High prediction instability detected")

if confidence_shift > 0.10:
    reasons.append("Significant confidence shift observed")

if feature_drift_score > 0.05:
    reasons.append("Model reasoning changed (feature drift)")

print("\nRisk Explanation:")
for r in reasons:
    print("-", r)
