import joblib
import numpy as np

# Load test data
X_test = joblib.load("data/X_test.pkl")

# Load models
model_v1 = joblib.load("models/model_v1.pkl")
model_v2 = joblib.load("models/model_v2.pkl")

# Predictions
preds_v1 = model_v1.predict(X_test)
preds_v2 = model_v2.predict(X_test)

# Prediction flip rate
prediction_flip_rate = np.mean(preds_v1 != preds_v2)
print("Prediction Flip Rate:", round(prediction_flip_rate, 4))

# Prediction probabilities
proba_v1 = model_v1.predict_proba(X_test)
proba_v2 = model_v2.predict_proba(X_test)

# Confidence shift
confidence_shift = np.mean(np.abs(proba_v1 - proba_v2))
print("Average Confidence Shift:", round(confidence_shift, 4))
