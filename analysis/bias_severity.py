import numpy as np
import pandas as pd

def compute_bias_severity(model, X_test, y_test, feature):
    df = X_test.copy()
    df["true"] = y_test.values
    df["group"] = pd.qcut(df[feature], 3, labels=["Low", "Medium", "High"])

    overall_acc = np.mean(model.predict(X_test) == y_test)
    subgroup_drops = []

    for g in ["Low", "Medium", "High"]:
        sub = df[df["group"] == g]
        if len(sub) < 10:
            continue
        acc = np.mean(model.predict(sub.drop(columns=["true", "group"])) == sub["true"])
        subgroup_drops.append(overall_acc - acc)

    return max(subgroup_drops)
