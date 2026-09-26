# ML Inference API Capstone

## Project Overview

This project deploys a trained machine learning model as a REST API using FastAPI.

The API predicts whether a customer is likely to churn based on customer information.

## Machine Learning Model

The project uses four machine learning models:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The models were tuned using GridSearchCV with Stratified K-Fold cross-validation.

The selected champion model is stored as:

`champion_model.joblib`

## API Technology

- Python
- FastAPI
- Uvicorn
- Scikit-Learn
- Pandas
- Joblib
- Pydantic

## API Endpoints

### GET /

Checks whether the API is running and whether the trained model is loaded.

### POST /predict

Accepts customer information and returns:

- Prediction
- Prediction label
- Probability of not churning
- Probability of churning

## Example Request

```json
{
  "tenure_months": 12,
  "support_tickets": 2,
  "monthly_spend_inr": 999,
  "last_login_days": 5,
  "plan_type": "Basic"
}
