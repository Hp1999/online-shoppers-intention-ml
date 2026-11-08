"""
Streamlit front-end for the Online Shoppers Purchasing Intention classifiers.

The app loads pre-trained pipelines from ``model/`` and scores an uploaded test
CSV. No model is retrained inside the app.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Online Shoppers Intention Classifier", layout="wide")


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(os.path.join(MODEL_DIR, file_name))


def positive_class_scores(estimator, features: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.decision_function(features)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # AUC needs both classes present in the uploaded labels.
    if len(np.unique(y_true)) > 1:
        metrics["AUC"] = roc_auc_score(y_true, y_score)
    else:
        metrics["AUC"] = float("nan")
    return metrics


st.title("🛒 Online Shoppers Purchasing Intention — Classifier Explorer")
st.write(
    "Upload a **test CSV** (must include the `Revenue` target column). "
    "A pre-trained model is loaded from disk and evaluated on your data — "
    "the model is **not** retrained."
)

# ---- Sidebar controls --------------------------------------------------------
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

overview_path = os.path.join(HERE, "metrics.csv")
if os.path.exists(overview_path):
    st.sidebar.markdown("### Reference metrics (held-out test set)")
    st.sidebar.dataframe(
        pd.read_csv(overview_path).round(4), use_container_width=True, hide_index=True
    )

# ---- Data upload -------------------------------------------------------------
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is None:
    st.info(
        "Awaiting a CSV upload. You can use the `test_data.csv` file bundled in "
        "the repository."
    )
    st.stop()

data = pd.read_csv(uploaded)
st.subheader("Preview of uploaded data")
st.dataframe(data.head(), use_container_width=True)

if TARGET not in data.columns:
    st.error(f"Uploaded CSV must contain the target column `{TARGET}`.")
    st.stop()

y_true = data[TARGET].astype(int)
x_features = data.drop(columns=[TARGET])

model = load_model(MODEL_FILES[model_name])

try:
    y_pred = model.predict(x_features)
    y_score = positive_class_scores(model, x_features)
except Exception as exc:  # feature mismatch / schema errors
    st.error(f"Could not score the data with `{model_name}`: {exc}")
    st.stop()

metrics = compute_metrics(y_true, y_pred, y_score)

# ---- Metrics display ---------------------------------------------------------
st.subheader(f"Evaluation metrics — {model_name}")
cols = st.columns(6)
ordered = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
for col, key in zip(cols, ordered):
    value = metrics[key]
    col.metric(key, "n/a" if np.isnan(value) else f"{value:.4f}")

# ---- Confusion matrix + classification report --------------------------------
left, right = st.columns(2)

with left:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Purchase (0)", "Purchase (1)"],
        yticklabels=["No Purchase (0)", "Purchase (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("#### Classification Report")
    report = classification_report(
        y_true, y_pred, target_names=["No Purchase (0)", "Purchase (1)"], output_dict=True
    )
    st.dataframe(
        pd.DataFrame(report).transpose().round(4), use_container_width=True
    )

st.caption(
    "Model loaded from disk and evaluated on the uploaded test set. "
    "No retraining is performed in the app."
)
