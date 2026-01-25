import joblib
import shap
import numpy as np
import pandas as pd

# Load test data
X_test = joblib.load("data/X_test.pkl")

# Load models
model_v1 = joblib.load("models/model_v1.pkl")
model_v2 = joblib.load("models/model_v2.pkl")

# Ensure X_test is a DataFrame with correct feature names
if not isinstance(X_test, pd.DataFrame):
    feature_names = model_v1.feature_names_in_
    X_test = pd.DataFrame(X_test, columns=feature_names)
else:
    feature_names = X_test.columns.to_list()

# Create SHAP explainers
explainer_v1 = shap.TreeExplainer(model_v1)
explainer_v2 = shap.TreeExplainer(model_v2)

# ---- SAFE SHAP VALUE EXTRACTION ----
def get_shap_values(explainer, X):
    shap_vals = explainer.shap_values(X)

    # Case 1: list (one array per class)
    if isinstance(shap_vals, list):
        return shap_vals[1]  # positive class

    # Case 2: single array (samples, features, classes)
    if shap_vals.ndim == 3:
        return shap_vals[:, :, 1]

    raise ValueError("Unexpected SHAP output format")

# Compute SHAP values safely
shap_v1 = get_shap_values(explainer_v1, X_test)
shap_v2 = get_shap_values(explainer_v2, X_test)

# Mean absolute SHAP values per feature
mean_shap_v1 = np.mean(np.abs(shap_v1), axis=0)
mean_shap_v2 = np.mean(np.abs(shap_v2), axis=0)

# Sanity check (VERY IMPORTANT)
assert len(mean_shap_v1) == len(feature_names), "Feature mismatch in v1"
assert len(mean_shap_v2) == len(feature_names), "Feature mismatch in v2"

# Feature importance drift score
feature_drift_score = np.mean(np.abs(mean_shap_v1 - mean_shap_v2))
print("Feature Importance Drift Score:", round(feature_drift_score, 6))

# Create drift DataFrame (NOW GUARANTEED SAFE)
drift_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance_v1": mean_shap_v1,
    "Importance_v2": mean_shap_v2,
    "Drift": np.abs(mean_shap_v1 - mean_shap_v2)
}).sort_values(by="Drift", ascending=False)

print("\nTop 10 Drifted Features:")
print(drift_df.head(10))
