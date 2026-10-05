# Credit Card Fraud Detection

A machine learning model that flags fraudulent credit card transactions
in real time, trained on a dataset of 284,807 real transactions with a
0.17% fraud rate.

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
without generating too many false alarms that would annoy legitimate
customers (high precision).

---

## The Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 30 features
- **Class imbalance:** 492 frauds (0.17%) vs. 284,315 normals (99.83%)
- **Features:** `Time`, `Amount`, and 28 anonymized PCA-transformed features (V1-V28)
- **Missing values:** None

The V-features are the result of PCA transformation applied by the dataset
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

---

## What I Learned

- **Accuracy is a trap** for imbalanced datasets. A model that predicts
  "never fraud" scores 99.83% accuracy and is completely useless.
- **AUPRC is the right metric** for imbalanced classification. It captures
  the model's performance across all thresholds, not just the default 0.5.
- **`scale_pos_weight` is essential.** Without it, XGBoost ignores the
  minority class because it's optimizing overall error rate.
- **The signal is weak.** The strongest single feature correlation is only
  -0.33. Fraud detection requires combining many weak signals, not finding
  one strong one.

---

## How to Run

**Prerequisites:**

- Python 3.10+
- Kaggle account (for dataset download)

**Setup:**

```bash
git clone https://github.com/gabrielmussafiri/card-fraude-detection
cd project-2-fraud-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
