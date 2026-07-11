"""
train_supervised.py
Trains 4 supervised models on the UCI Heart Disease dataset to predict
the *probability* (chance %) of heart disease, and saves:
  - each trained model (.pkl)
  - the fitted StandardScaler (.pkl)
  - metrics.json (accuracy, precision, recall, f1, confusion matrix per model)

Run: python3 train_supervised.py
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

DATA_PATH = "data/heart.csv"
MODELS_DIR = "models"

# Order matters! The Flask backend + frontend form must send features in this
# exact order at inference time.
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_ORDER]
    y = df["target"]

    # 2. Stratified 80/20 split (stratify keeps the disease/no-disease ratio
    # the same in both train and test sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scale features (fit ONLY on train, then apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Define models. probability=True on SVC is required for predict_proba.
    models = {
        "linear_regression": LinearRegression(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "svm": SVC(probability=True, kernel="rbf", random_state=42),
    }

    metrics = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if name == "linear_regression":
            # Threshold continuous predictions at 0.5 for classification metrics evaluation
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            y_pred_binary = y_pred

        acc = accuracy_score(y_test, y_pred_binary)
        prec = precision_score(y_test, y_pred_binary)
        rec = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        cm = confusion_matrix(y_test, y_pred_binary).tolist()

        metrics[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        }

        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")
        print(f"{name:15s} | acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

    # 5. Save scaler + feature order + metrics
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{MODELS_DIR}/feature_order.json", "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)

    print("\nAll models, scaler, and metrics saved to models/")


if __name__ == "__main__":
    main()
