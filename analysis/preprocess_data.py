import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path="data/wdbc.data"):
    """
    Loads and preprocesses the Breast Cancer Wisconsin dataset
    for Automated Model Update Risk Assessment System.
    """

    # Column names from wdbc.names
    columns = [
        "id", "diagnosis",
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "mean_smoothness", "mean_compactness", "mean_concavity",
        "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
        "radius_error", "texture_error", "perimeter_error", "area_error",
        "smoothness_error", "compactness_error", "concavity_error",
        "concave_points_error", "symmetry_error", "fractal_dimension_error",
        "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
        "worst_smoothness", "worst_compactness", "worst_concavity",
        "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
    ]

    # Load dataset
    df = pd.read_csv(file_path, header=None, names=columns)

    # Convert target labels (M = 1, B = 0)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Drop ID column
    df.drop(columns=["id"], inplace=True)

    # Separate features and target
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Handle missing values (safety step)
    X = X.fillna(X.mean())

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Save processed data for reuse across project
    joblib.dump(X_train, "data/X_train.pkl")
    joblib.dump(X_test, "data/X_test.pkl")
    joblib.dump(y_train, "data/y_train.pkl")
    joblib.dump(y_test, "data/y_test.pkl")

    print("Preprocessing completed successfully")
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    return X_train, X_test, y_train, y_test
