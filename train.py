"""
Train five classifiers on the UCI Online Shoppers Purchasing Intention dataset
and persist each fitted pipeline so the Streamlit app can score uploaded test
data without retraining.

Dataset: https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset
Target  : Revenue (True -> purchase session, False -> no purchase) [binary]
"""

import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier  # noqa: F401  (kept off; reference only)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00468/online_shoppers_intention.csv"
)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "model")
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["Month", "VisitorType", "Weekend"]
TARGET = "Revenue"


def load_dataset() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    local_csv = os.path.join(DATA_DIR, "online_shoppers_intention.csv")
    if not os.path.exists(local_csv):
        print(f"Downloading dataset -> {local_csv}")
        urllib.request.urlretrieve(DATA_URL, local_csv)
    frame = pd.read_csv(local_csv)
    print(f"Loaded dataset: {frame.shape[0]} rows, {frame.shape[1] - 1} features")
    return frame


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        col for col in feature_frame.columns if col not in CATEGORICAL_FEATURES
    ]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def positive_class_scores(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return a continuous score for the positive class for AUC computation."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.decision_function(features)


def evaluate(estimator: Pipeline, features: pd.DataFrame, labels: np.ndarray) -> dict:
    predictions = estimator.predict(features)
    scores = positive_class_scores(estimator, features)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(labels, scores),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    frame = load_dataset()

    labels = frame[TARGET].astype(int)
    features = frame.drop(columns=[TARGET])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Persist the held-out test split (raw features + target) for the Streamlit app.
    test_export = x_test.copy()
    test_export[TARGET] = y_test.values
    test_export.to_csv(os.path.join(HERE, "test_data.csv"), index=False)
    print(f"Saved test_data.csv with {len(test_export)} rows")

    preprocessor = build_preprocessor(features)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    }

    file_slugs = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Support Vector Machine": "svm",
    }

    rows = []
    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[("preprocess", preprocessor), ("model", classifier)]
        )
        pipeline.fit(x_train, y_train)
        metrics = evaluate(pipeline, x_test, y_test)
        metrics["ML Model Name"] = name
        rows.append(metrics)

        out_path = os.path.join(MODEL_DIR, f"{file_slugs[name]}.joblib")
        joblib.dump(pipeline, out_path)
        print(f"Trained + saved: {name:24s} -> {out_path}")

    metrics_frame = pd.DataFrame(rows)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    metrics_frame.to_csv(os.path.join(HERE, "metrics.csv"), index=False)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Comparison Table (test set) ===")
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
