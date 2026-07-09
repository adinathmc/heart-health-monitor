"""
app.py - Flask backend for the Heart Disease Risk Predictor.
No authentication - simple, open API for the college demo.

Endpoints:
  POST /predict       -> chance % of heart disease from each of 4 models
  GET  /metrics        -> accuracy/precision/recall/f1 per model
  POST /cluster        -> which risk cluster a patient falls into
  GET  /cluster-data    -> all patients' 2D PCA points (for scatter plot)
  GET  /health          -> simple health check
"""

import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the frontend (served separately) to call this API

MODELS_DIR = "models"

# ---- Load everything once at startup ----
with open(f"{MODELS_DIR}/feature_order.json") as f:
    FEATURE_ORDER = json.load(f)

scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")

supervised_models = {
    "logistic": joblib.load(f"{MODELS_DIR}/logistic.pkl"),
    "knn": joblib.load(f"{MODELS_DIR}/knn.pkl"),
    "decision_tree": joblib.load(f"{MODELS_DIR}/decision_tree.pkl"),
    "svm": joblib.load(f"{MODELS_DIR}/svm.pkl"),
}

with open(f"{MODELS_DIR}/metrics.json") as f:
    METRICS = json.load(f)

unsupervised_scaler = joblib.load(f"{MODELS_DIR}/unsupervised_scaler.pkl")
kmeans = joblib.load(f"{MODELS_DIR}/kmeans.pkl")
pca = joblib.load(f"{MODELS_DIR}/pca.pkl")

with open(f"{MODELS_DIR}/cluster_summary.json") as f:
    CLUSTER_SUMMARY = json.load(f)

with open(f"{MODELS_DIR}/cluster_points.json") as f:
    CLUSTER_POINTS = json.load(f)


def bucket(pct):
    """Map a 0-100 percentage to a risk label."""
    if pct < 34:
        return "Low chance"
    elif pct < 67:
        return "Moderate chance"
    else:
        return "High chance"


def extract_features(data):
    """Pull features from the incoming JSON in the correct fixed order."""
    try:
        return np.array([[float(data[feat]) for feat in FEATURE_ORDER]])
    except KeyError as e:
        raise ValueError(f"Missing required field: {e}")
    except (TypeError, ValueError):
        raise ValueError("All fields must be numeric")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        features = extract_features(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    scaled = scaler.transform(features)

    results = {}
    total_chance = 0.0

    for name, model in supervised_models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled)[0][1] * 100  # % chance of class 1
        else:
            # Linear Regression outputs continuous target value. Clip and scale to 0-100
            pred_val = model.predict(scaled)[0]
            prob = float(np.clip(pred_val * 100, 0, 100))
            
        results[name] = {
            "chance_percent": round(float(prob), 1),
            "bucket": bucket(prob),
        }
        total_chance += prob

    overall = round(total_chance / len(supervised_models), 1)

    return jsonify({
        "models": results,
        "overall_chance_percent": overall,
        "overall_bucket": bucket(overall),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(METRICS)


@app.route("/cluster", methods=["POST"])
def cluster():
    data = request.get_json(force=True)

    try:
        features = extract_features(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    scaled = unsupervised_scaler.transform(features)
    cluster_id = int(kmeans.predict(scaled)[0])
    point_2d = pca.transform(scaled)[0].tolist()

    return jsonify({
        "cluster": cluster_id,
        "cluster_label": CLUSTER_SUMMARY[str(cluster_id)]["label"],
        "cluster_avg_rate_percent": CLUSTER_SUMMARY[str(cluster_id)]["avg_disease_rate_percent"],
        "point": {"x": round(point_2d[0], 3), "y": round(point_2d[1], 3)},
    })


@app.route("/cluster-data", methods=["GET"])
def cluster_data():
    return jsonify({
        "points": CLUSTER_POINTS,
        "summary": CLUSTER_SUMMARY,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
