libraries = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "shap",
    "joblib",
    "streamlit"
]

for lib in libraries:
    try:
        __import__(lib)
        print(f"{lib}: INSTALLED ✅")
    except ImportError:
        print(f"{lib}: NOT INSTALLED ❌")
