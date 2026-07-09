# Heart Health Monitor

A heart disease risk predictor combining **supervised learning** (chance % prediction)
and **unsupervised learning** (patient clustering), built with Flask + vanilla JS.

## Project Structure

```
heart-health-monitor/
├── model_training/
│   ├── data/heart.csv              # UCI Heart Disease dataset (297 patients)
│   ├── train_supervised.py         # Trains Logistic Regression, KNN, Decision Tree, SVM
│   ├── train_unsupervised.py       # K-Means (k=3) + PCA clustering
│   └── models/                     # All trained models + JSON artifacts (generated)
├── backend/
│   ├── app.py                      # Flask API (no auth)
│   ├── requirements.txt
│   └── models/                     # Copy of trained models used by the API
└── frontend/
    ├── index.html                  # Single-page app (Predict + Clusters tabs)
    ├── style.css
    └── script.js
```

## How to Run

### 1. (Optional) Retrain the models
Only needed if you change the dataset or want to re-tune models.

```bash
cd model_training
pip install pandas numpy scikit-learn joblib
python3 train_supervised.py
python3 train_unsupervised.py
cp models/*.pkl models/*.json ../backend/models/
```

### 2. Start the Flask backend

```bash
cd backend
pip install -r requirements.txt
python3 app.py
```
This runs on **http://127.0.0.1:5000**. Keep this terminal open.

### 3. Serve the frontend (separate terminal)

Browsers block `fetch()` calls from `file://` pages, so serve the frontend folder
with a simple static server instead of double-clicking index.html:

```bash
cd frontend
python3 -m http.server 8080
```

Then open **http://127.0.0.1:8080** in your browser.

## API Endpoints (Flask, no auth)

| Route | Method | Purpose |
|---|---|---|
| `/predict` | POST | Returns chance % of heart disease from 4 models + overall average |
| `/metrics` | GET | Accuracy/precision/recall/F1 for each model |
| `/cluster` | POST | Which risk cluster (Low/Moderate/High) a patient falls into |
| `/cluster-data` | GET | All 297 patients' 2D PCA coordinates + cluster labels |
| `/health` | GET | Simple health check |

## Model Performance (on 20% held-out test set)

| Model | Accuracy |
|---|---|
| SVM | ~85% |
| Logistic Regression | ~83% |
| KNN | ~83% |
| Decision Tree | ~75% |

## Notes for Presentation

- The dataset is the classic **UCI Cleveland Heart Disease dataset** (297 patients, 13 features).
- Supervised models answer **"what is the chance?"** — a probability from 0-100%.
- Unsupervised K-Means answers **"what natural groups exist?"** — found without ever
  seeing the diagnosis label, then labeled after the fact by checking each cluster's
  actual disease rate (Low ~10%, Moderate ~32%, High ~90%).
- This dual view (prediction vs. pattern discovery) is the core narrative: supervised
  learning needs labeled data to predict, unsupervised learning finds structure without it.

**Disclaimer:** This is an academic project, not a medical diagnostic tool.
