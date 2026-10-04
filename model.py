import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score , precision_score, recall_score , average_precision_score
import mlflow


# Import Data
df = pd.read_csv('./data/creditcard.csv')

mlflow.set_experiment("MLflow Fraud Detection Experiment")

# Split the data into features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

print(X.columns.tolist())
print(X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate the scale_pos_weight parameter for XGBoost to handle class imbalance
count_normal = y_train.value_counts()[0]
count_fraud = y_train.value_counts()[1]
scale_pos_weight = (count_normal / count_fraud)

with mlflow.start_run():
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Train the model
    
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight, 
        random_state=42,
        eval_metric='aucpr', 
        )
    
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
  
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    auprc = average_precision_score(y_test, y_proba)
   
    
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification_rep)
    print(f"AUPRC: {auprc:.4f}")
    
    mlflow.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auprc": auprc
    })
    mlflow.xgboost.log_model(model,name = "xgboost_model")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    