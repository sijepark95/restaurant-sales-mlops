from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["alias"] == "champion"


def test_prediction_endpoint() -> None:
    request_data = {
        "prediction_date": "2026-07-25",
        "temperature": 82,
        "promotion": True,
    }

    response = client.post("/predict", json=request_data)
    result = response.json()

    assert response.status_code == 200
    assert result["prediction_date"] == "2026-07-25"
    assert result["predicted_sales"] > 0
    assert result["model_name"] == "restaurant-sales-forecaster"
    assert result["model_alias"] == "champion"


def test_invalid_temperature_is_rejected() -> None:
    request_data = {
        "prediction_date": "2026-07-25",
        "temperature": 200,
        "promotion": False,
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 422


def test_missing_promotion_is_rejected() -> None:
    request_data = {
        "prediction_date": "2026-07-25",
        "temperature": 82,
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 422