import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================
# Load model & threshold
# ==========================================

model = joblib.load("models/model_v1.pkl")

with open("models/threshold.json", "r") as f:
    threshold_data = json.load(f)

threshold = threshold_data["best_threshold"]

# ==========================================
# Create FastAPI app
# ==========================================

app = FastAPI(title="Loan Default Prediction API")


# ==========================================s
# Define Input Schema
# ==========================================

class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_exp: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float
    previous_loan_defaults_on_file: str
    credit_score: float
    person_gender: str
    person_education: str


# ==========================================
# Health Check
# ==========================================

@app.get("/")
def health():
    return {"message": "Loan Default API is running 🚀"}


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict(application: LoanApplication):

    # Convert input to dataframe
    input_data = pd.DataFrame([application.dict()])

    # Get probability
    probability = model.predict_proba(input_data)[0][1]

    # Apply threshold
    prediction = int(probability >= threshold)

    return {
        "probability_of_default": round(float(probability), 4),
        "threshold_used": threshold,
        "prediction": prediction,
        "decision": "Reject Loan ❌" if prediction == 1 else "Approve Loan ✅"
    }
