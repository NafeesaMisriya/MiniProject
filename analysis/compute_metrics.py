import shap
import numpy as np
import pandas as pd

def compute_metrics(model_v1, model_v2, X_test):
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=model_v1.feature_names_in_)

    # Prediction flip rate
    flip_rate = np.mean(
        model_v1.predict(X_test) != model_v2.predict(X_test)
    )

    # Confidence shift
    conf_shift = np.mean(
        np.abs(
            model_v1.predict_proba(X_test) -
            model_v2.predict_proba(X_test)
        )
    )

    # SHAP Feature Drift
    expl1 = shap.TreeExplainer(model_v1)
    expl2 = shap.TreeExplainer(model_v2)

    def extract_shap(expl):
        sv = expl.shap_values(X_test)
        if isinstance(sv, list):
            return sv[1]
        return sv[:, :, 1]

    s1 = extract_shap(expl1)
    s2 = extract_shap(expl2)

    drift = np.mean(np.abs(
        np.mean(np.abs(s1), axis=0) -
        np.mean(np.abs(s2), axis=0)
    ))

    return flip_rate, conf_shift, drift
