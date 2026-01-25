import pandas as pd
import numpy as np

def compute_subgroup_risk(model_v1, model_v2, X_test, y_test, feature):
    df = X_test.copy()
    df["true"] = y_test.values

    df["group"] = pd.qcut(df[feature], 3, labels=["Low", "Medium", "High"])

    max_drop = 0
    for g in ["Low", "Medium", "High"]:
        sub = df[df["group"] == g]
        if len(sub) < 10:
            continue

        Xg = sub.drop(columns=["true", "group"])
        acc1 = np.mean(model_v1.predict(Xg) == sub["true"])
        acc2 = np.mean(model_v2.predict(Xg) == sub["true"])

        max_drop = max(max_drop, acc1 - acc2)

    return max_drop
