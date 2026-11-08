# Online Shoppers Purchasing Intention — Classification & Streamlit App

End-to-end machine-learning assignment: five classification models are trained
on a single dataset, evaluated with six metrics, and served through an
interactive Streamlit web app deployed on Streamlit Community Cloud.

---

## a. Problem Statement

Predict whether an online shopping session will end in a **purchase**
(`Revenue = True`) or **not** (`Revenue = False`). This is a **binary
classification** problem. Accurately identifying purchase-intent sessions lets
e-commerce platforms target promotions, improve conversion, and reduce cart
abandonment.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository — *Online Shoppers Purchasing
  Intention Dataset*
  (https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset)
- **Instances:** 12,330 sessions (**> 500 required** ✔)
- **Features:** 17 input features (**> 12 required** ✔) + 1 target (`Revenue`)
- **Target:** `Revenue` (boolean) — purchase vs. no-purchase. The classes are
  imbalanced (~15.5% purchases), which makes AUC, F1 and MCC more informative
  than raw accuracy.
- **Feature types:**
  - Numeric (14): `Administrative`, `Administrative_Duration`, `Informational`,
    `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`,
    `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`,
    `OperatingSystems`, `Browser`, `Region`, `TrafficType`
  - Categorical (3): `Month`, `VisitorType`, `Weekend`
- **Preprocessing:** a single `ColumnTransformer` applies `StandardScaler` to
  numeric features and `OneHotEncoder(handle_unknown="ignore")` to categorical
  features. It is wrapped together with each classifier inside a scikit-learn
  `Pipeline`, so every saved model preprocesses raw input automatically.
- A stratified 75/25 train/test split is used; the 25% hold-out is exported as
  `test_data.csv` for the Streamlit app.

## c. GitHub Repository Link

> https://github.com/Hp1999/online-shoppers-intention-ml

The repository contains:

```
online-shoppers-intention-ml/
├── app.py                 # Streamlit application
├── train.py               # Trains + saves all five models and metrics
├── requirements.txt       # Dependencies
├── README.md              # This file
├── test_data.csv          # Hold-out test set (upload this in the app)
├── metrics.csv            # Saved comparison metrics
├── data/
│   └── online_shoppers_intention.csv   # Raw dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## d. Models Used

Five classifiers trained on the **same** dataset with identical preprocessing:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (k = 7)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (RBF kernel)

### Comparison Table (hold-out test set, 3,083 sessions)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.8780   | 0.8866 | 0.7149    | 0.3522 | 0.4719 | 0.4450 |
| Decision Tree          | 0.8557   | 0.7262 | 0.5332    | 0.5388 | 0.5360 | 0.4505 |
| KNN                    | 0.8709   | 0.7972 | 0.6479    | 0.3627 | 0.4651 | 0.4200 |
| Naive Bayes            | 0.6786   | 0.7973 | 0.2976    | 0.7925 | 0.4327 | 0.3308 |
| Support Vector Machine | 0.8855   | 0.8664 | 0.7000    | 0.4549 | 0.5515 | 0.5041 |

### Observations

| ML Model Name          | Observation about model performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Highest AUC (0.8866) and strong precision, so its positive predictions are trustworthy, but low recall (0.35) means it misses most true buyers under the default threshold — typical of a linear model on an imbalanced set. |
| Decision Tree          | The most balanced precision/recall pair, giving a solid F1 (0.536); however the lowest AUC (0.726) shows weak probability ranking and a tendency to overfit sharp splits. |
| KNN                    | Middle-of-the-pack accuracy with precision favoured over recall; distance-based voting struggles on the high-dimensional one-hot space and with class imbalance. |
| Naive Bayes            | Highest recall (0.79) — it flags most buyers — but its feature-independence assumption breaks on correlated page-duration features, collapsing precision (0.30) and overall accuracy. Useful only when catching buyers matters more than false alarms. |
| Support Vector Machine | Best overall: top accuracy (0.8855), best F1 (0.5515) and best MCC (0.5041), with AUC close to the leader. The RBF kernel captures non-linear interactions and handles the imbalance best. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** — it wins on Accuracy, F1 and MCC, and is a close second on AUC, making it the most reliable single model for this dataset. (Logistic Regression is the alternative if pure AUC / ranking quality is the priority.) |

---

## Streamlit App

The deployed app lets you:

- **Upload a test CSV** (use `test_data.csv`; it must include the `Revenue`
  target column and all input features).
- **Select a model** from a dropdown (all five classifiers).
- **View the six evaluation metrics** computed on your uploaded data.
- **Inspect a confusion matrix** and a full classification report.

The app loads the saved `.joblib` pipeline and scores the uploaded data
directly — **it never retrains the model**.

### Live App Link

> https://<your-app-name>.streamlit.app
>
> *(Replace with your actual Streamlit Community Cloud URL after deploying.)*
> It will look like `https://online-shoppers-intention-ml.streamlit.app` or
> similar, depending on the name Streamlit assigns at deploy time.

---

## How to Run Locally

```bash
pip install -r requirements.txt
python train.py        # (optional) regenerates models + metrics + test_data.csv
streamlit run app.py
```

## How to Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App**, select this repository and the `main` branch.
4. Set the main file to `app.py` and click **Deploy**.
5. Open the generated link and upload `test_data.csv` to see results.
