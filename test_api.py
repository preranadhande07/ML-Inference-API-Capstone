from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()
    assert "model_loaded" in response.json()


def test_predict_endpoint():
    data = {
        "tenure_months": 12,
        "support_tickets": 2,
        "monthly_spend_inr": 999,
        "last_login_days": 5,
        "plan_type": "Basic"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "prediction_label" in result
    assert "probability_not_churn" in result
    assert "probability_churn" in result


def test_invalid_request():
    data = {
        "tenure_months": 12,
        "support_tickets": 2
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422