"""
train_unsupervised.py
Runs K-Means clustering (k=2) and PCA (2D) on the same scaled heart disease
features (WITHOUT the target column - unsupervised never sees labels).

After clustering, we peek at the actual target values only to LABEL each
cluster (e.g. "Low chance") for interpretability - the clustering itself
never used the target.

Saves:
  - kmeans.pkl, pca.pkl
  - cluster_points.json  -> every patient's 2D PCA point + cluster id +
                            cluster risk label (for the frontend scatter plot)
  - cluster_summary.json -> per-cluster average disease rate + label + size
  - elbow_data.json       -> inertia for k=1..8, to justify k=2 in your slides

Run: python3 train_unsupervised.py
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH = "data/heart.csv"
MODELS_DIR = "models"

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

K = 2  # fixed cluster count


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_ORDER]
    y = df["target"]  # only used AFTER clustering, to label clusters

    # 1. Scale features - reuse the SAME scaling approach as supervised
    # (fit fresh here since unsupervised training can use the full dataset,
    # not just a train split - there's no leakage risk without labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Elbow method data (k=1 to 8) - for justifying k=2 in your slides
    elbow_data = []
    for k in range(1, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        elbow_data.append({"k": k, "inertia": round(km.inertia_, 2)})

    with open(f"{MODELS_DIR}/elbow_data.json", "w") as f:
        json.dump(elbow_data, f, indent=2)

    # 3. Fit final KMeans with k=2
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # 4. Fit PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    points_2d = pca.fit_transform(X_scaled)

    # 5. Label each cluster by its average disease rate (peek at y here only).
    # We RANK the 2 clusters relative to each other (lowest -> Low chance,
    # highest -> High chance) rather than using fixed percentage
    # thresholds. This guarantees a clean 2-way Low/High story
    # regardless of where the absolute rates happen to fall.
    raw_rates = {}
    for cid in range(K):
        mask = cluster_ids == cid
        raw_rates[cid] = float(y[mask].mean())

    ranked_cids = sorted(raw_rates, key=lambda c: raw_rates[c])  # lowest -> highest
    rank_labels = ["Low chance", "High chance"]

    cluster_summary = {}
    for rank, cid in enumerate(ranked_cids):
        mask = cluster_ids == cid
        cluster_summary[str(cid)] = {
            "avg_disease_rate_percent": round(raw_rates[cid] * 100, 1),
            "label": rank_labels[rank],
            "size": int(mask.sum()),
        }

    with open(f"{MODELS_DIR}/cluster_summary.json", "w") as f:
        json.dump(cluster_summary, f, indent=2)

    # 6. Save every patient's 2D point + cluster + label for the scatter plot
    cluster_points = []
    for i in range(len(df)):
        cid = int(cluster_ids[i])
        cluster_points.append({
            "x": round(float(points_2d[i][0]), 3),
            "y": round(float(points_2d[i][1]), 3),
            "cluster": cid,
            "cluster_label": cluster_summary[str(cid)]["label"],
            "actual_target": int(y.iloc[i]),
        })

    with open(f"{MODELS_DIR}/cluster_points.json", "w") as f:
        json.dump(cluster_points, f, indent=2)

    # 7. Save models
    joblib.dump(kmeans, f"{MODELS_DIR}/kmeans.pkl")
    joblib.dump(pca, f"{MODELS_DIR}/pca.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/unsupervised_scaler.pkl")

    print("Cluster summary:")
    for cid, info in cluster_summary.items():
        print(f"  Cluster {cid}: {info['label']} "
              f"(avg {info['avg_disease_rate_percent']}%, {info['size']} patients)")

    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print("All unsupervised artifacts saved to models/")


if __name__ == "__main__":
    main()