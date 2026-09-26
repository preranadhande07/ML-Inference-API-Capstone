from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="ML Inference API",
    description="REST API for Customer Churn Prediction",
    version="1.0.0"
)

MODEL_PATH = "champion_model.joblib"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} was not found.")

    model = joblib.load(MODEL_PATH)
    print("Champion model loaded successfully!")

except Exception as e:
    model = None
    print(f"Model loading error: {e}")


class CustomerData(BaseModel):
    tenure_months: float
    support_tickets: float
    monthly_spend_inr: float
    last_login_days: float
    plan_type: str


@app.get("/")
def root():
    return {
        "message": "ML Inference API is running",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: CustomerData):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Champion model could not be loaded."
        )

    input_data = pd.DataFrame([{
        "tenure_months": data.tenure_months,
        "support_tickets": data.support_tickets,
        "monthly_spend_inr": data.monthly_spend_inr,
        "last_login_days": data.last_login_days,
        "plan_type": data.plan_type
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        return {
            "prediction": prediction,
            "prediction_label": (
                "Churn" if prediction == 1 else "Not Churn"
            ),
            "probability_not_churn": float(probabilities[0]),
            "probability_churn": float(probabilities[1])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )