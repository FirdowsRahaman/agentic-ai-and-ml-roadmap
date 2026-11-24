from fastapi.testclient import TestClient
from src.app import app


def test_health_and_prediction_and_drift():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}

        pred_resp = client.post(
            "/predict",
            json={"tenure_months": 3.0, "monthly_spend": 120.0, "support_calls": 5},
        )
        assert pred_resp.status_code == 200
        body = pred_resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0

        # Heavily shifted spend distribution should trigger PSI >= 0.25
        drifted = client.post("/drift", json={"recent_monthly_spends": [195.0] * 40})
        assert drifted.status_code == 200
        assert drifted.json()["retrain_required"] is True
