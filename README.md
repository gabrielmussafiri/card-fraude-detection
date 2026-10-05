# Credit Card Fraud Detection

A machine learning model that flags fraudulent credit card transactions in
real time. Trained on 284,807 real transactions with a 0.17% fraud rate,
served via a FastAPI endpoint, and deployed as a Docker container.

---

## Results

| Metric            | Score  |
| ----------------- | ------ |
| Recall (fraud)    | 0.83   |
| Precision (fraud) | 0.87   |
| F1 (fraud)        | 0.85   |
| AUPRC             | 0.8791 |

**What this means:** The model catches 83% of fraudulent transactions
(4 out of 5 frauds) while maintaining 87% precision — when the model
flags a transaction as fraud, it's correct 87% of the time. The AUPRC
of 0.88 indicates strong performance across all decision thresholds.

**Confusion matrix (test set, 56,962 transactions):**

- True Negatives: 56,852
- False Positives: 12
- False Negatives: 17
- True Positives: 81

---

## The Problem

Credit card fraud costs the global economy an estimated $32 billion
annually. The challenge: fraud is extremely rare (0.17% of transactions
in this dataset), so a naive model that predicts "never fraud" achieves
99.83% accuracy while catching zero frauds.

The goal is a model that catches as much fraud as possible (high recall)
without generating too many false alarms (high precision).

---

## The Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 30 features
- **Class imbalance:** 492 frauds (0.17%) vs. 284,315 normals (99.83%)
- **Features:** `Time`, `Amount`, and 28 anonymized PCA-transformed features (V1–V28)
- **Missing values:** None

The V-features are the result of a PCA transformation applied by the dataset
authors to protect user privacy. They retain the signal but not the
original meaning.

---

## Approach

1. **Load data** from CSV into pandas.
2. **Split** into train (80%) and test (20%) with `stratify=y` to preserve
   the class ratio.
3. **Handle imbalance** with `scale_pos_weight` in XGBoost (set to the ratio
   of negative to positive samples in the training set).
4. **Train** an XGBoost classifier with `eval_metric='aucpr'`.
5. **Evaluate** with precision, recall, F1, AUPRC, confusion matrix, and
   classification report.
6. **Track** all experiments in MLflow for reproducibility and comparison.
7. **Serve** the trained model via a FastAPI endpoint.
8. **Package** the API as a Docker container for reproducible deployment.

---

## What I Learned

- **Accuracy is a trap** for imbalanced datasets. A model that predicts
  "never fraud" scores 99.83% accuracy and is completely useless.
- **AUPRC is the right metric** for imbalanced classification. It captures
  the model's performance across all thresholds, not just the default 0.5.
- **`scale_pos_weight` is essential.** Without it, XGBoost ignores the
  minority class because it's optimizing overall error rate.
- **Scaling isn't always necessary.** XGBoost is invariant to monotonic
  transformations of features. I tested with and without `StandardScaler`
  on `Amount` and `Time` — AUPRC was identical (0.8791), so I removed it
  for simplicity.
- **The signal is weak.** The strongest single feature correlation is only
  -0.33. Fraud detection requires combining many weak signals, not finding
  one strong one.

---

## How to Run

### Option 1: Docker (recommended)

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

Then open `http://localhost:8000/docs` for the interactive API.

### Option 2: Local Python

**Prerequisites:**

- Python 3.13+
- Kaggle account (for dataset download)

**Setup:**

```bash
git clone https://github.com/gabrielmussafiri/card-fraude-detection.git
cd project-2-fraud-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Download the dataset:**

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
rm creditcardfraud.zip
```

**Run the EDA:**

```bash
python explore.py
```

**Train the model:**

```bash
python model.py
```

**View experiments:**

```bash
mlflow ui
```

Then open `http://localhost:5000`.

**Run the API locally:**

```bash
uvicorn app:app --reload
```

Then open `http://localhost:8000/docs`.

---

## API Reference

### `GET /health`

Returns `{"status": "ok"}` if the service is running.

### `POST /predict`

Accepts a transaction with 30 features and returns a prediction.

**Request body:**

```json
{
  "Time": 406,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "...": "...",
  "V28": -0.0210530534538215,
  "Amount": 149.62
}
```

**Response:**

```json
{
  "is_fraud": true,
  "probability": 0.9996
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406, "V1": -1.36, "V2": -0.07, "Amount": 149.62}'
```

---

## Tech Stack

- **Python 3.13**
- **pandas, numpy** — Data manipulation
- **scikit-learn** — Preprocessing, metrics, train/test split
- **XGBoost** — Gradient boosting classifier
- **MLflow** — Experiment tracking and model registry
- **FastAPI** — REST API for model serving
- **Uvicorn** — ASGI server
- **Docker** — Containerized deployment

---

## Project Structure

```
project-2-fraud-detection/
├── app/
│   └── app.py                 # FastAPI service
├── data/
│   └── creditcard.csv         # Dataset (gitignored)
├── explore.py                 # Exploratory data analysis
├── model.py                   # Model training and evaluation
├── model.pkl                  # Trained model artifact (gitignored)
├── Dockerfile                 # Container definition
├── .dockerignore              # Files excluded from Docker build
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes venv, data, .env, mlruns
└── README.md                  # This file
```

---

## Future Work

- Hyperparameter tuning (grid search or Optuna)
- Threshold tuning based on business cost trade-offs
- Try LightGBM and compare AUPRC
- Add model monitoring for data drift
- Set up CI/CD to auto-build and push the Docker image

---

## Repository

- **Best model run ID:** ``
- **MLflow experiment:** `"MLflow Fraud Detection Experiment"`
