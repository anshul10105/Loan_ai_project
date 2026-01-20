from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

# ---- Load model safely ----
MODEL_PATH = os.path.join("model", "loan_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---- Input schema ----
class LoanInput(BaseModel):
    income: float
    credit_score: int
    loan_amount: float

@app.get("/")
def home():
    return {"status": "Loan Prediction API is running"}

@app.post("/predict")
def predict_loan(data: LoanInput):
    try:
        X = [[
            float(data.income),
            float(data.loan_amount),
            int(data.credit_score)
        ]]

        proba = model.predict_proba(X)

        # Safe probability extraction
        if len(proba[0]) < 2:
            prediction_proba = 0.0
        else:
            prediction_proba = float(proba[0][1])

        return {
            "loan_approved": prediction_proba > 0.3,
            "confidence": round(prediction_proba, 2)
        }

    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__
        }






