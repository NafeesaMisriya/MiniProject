import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
X_train = joblib.load("data/X_train.pkl")
X_test = joblib.load("data/X_test.pkl")
y_train = joblib.load("data/y_train.pkl")
y_test = joblib.load("data/y_test.pkl")

# =========================
# CREATE SUBGROUPS
# =========================
def add_subgroups(X, y, feature):
    df = X.copy()
    df["target"] = y.values
    df["group"] = pd.qcut(df[feature], 3, labels=["Low", "Medium", "High"])
    return df

df_train = add_subgroups(X_train, y_train, "mean radius")

# =========================
# BASELINE MODEL v0 (DEPLOYED)
# =========================
model_v0 = RandomForestClassifier(
    n_estimators=120,
    max_depth=None,
    random_state=0
)
model_v0.fit(X_train, y_train)
joblib.dump(model_v0, "models/model_v0.pkl")

# =========================
# UPDATED MODEL v1 (SMALL CHANGE)
# - slightly different depth
# - small noise injection
# =========================
X_v1 = X_train.copy()
noise = np.random.normal(0, 0.02, X_v1.shape)
X_v1 = X_v1 + noise

model_v1 = RandomForestClassifier(
    n_estimators=120,
    max_depth=10,
    random_state=1
)
model_v1.fit(X_v1, y_train)
joblib.dump(model_v1, "models/model_v1.pkl")

# =========================
# UPDATED MODEL v2 (MODERATE BIAS)
# =========================
df_v2 = df_train[df_train["group"] != "High"]
X_v2 = df_v2.drop(columns=["target", "group"])
y_v2 = df_v2["target"]

model_v2 = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=2
)
model_v2.fit(X_v2, y_v2)
joblib.dump(model_v2, "models/model_v2.pkl")

# =========================
# UPDATED MODEL v3 (SEVERE BIAS)
# =========================
df_v3 = df_train[df_train["group"] != "Low"].copy()
noise_idx = df_v3.sample(frac=0.25, random_state=42).index
df_v3.loc[noise_idx, "target"] = 1 - df_v3.loc[noise_idx, "target"]

X_v3 = df_v3.drop(columns=["target", "group"])
y_v3 = df_v3["target"]

model_v3 = RandomForestClassifier(
    n_estimators=60,
    max_depth=4,
    random_state=3
)
model_v3.fit(X_v3, y_v3)
joblib.dump(model_v3, "models/model_v3.pkl")

print("✅ v0, v1, v2, v3 models trained successfully")
