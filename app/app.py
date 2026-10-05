import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title =' Fraud Detection API')

# Feature names based on the dataset
FEATURE_ORDER = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Load model at startup

model = joblib.load('model.pkl')

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
@app.get('/health')
def health_check():
    return {"status": "healthy"}

@app.post('/predict')
def predict(transaction: Transaction):
    # Convert request to DataFrame in the exact training order
    df =pd.DataFrame([transaction.model_dump()])[FEATURE_ORDER]
    
    # Predict
    prediction =int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    
    return{
        'is_fraud': bool(prediction),
        'probability': probability
    }