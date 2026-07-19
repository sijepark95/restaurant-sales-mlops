from datetime import date
import os
import mlflow
import pandas as pd
from pydantic import BaseModel, Field
from time import perf_counter
from fastapi import FastAPI, Response

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "restaurant-sales-forecaster"
MODEL_ALIAS = "champion"
MODEL_URI = os.getenv(
    "MODEL_URI",
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
)

FEATURES = [
    "day_of_week",
    "is_weekend",
    "temperature",
    "promotion",
    "month",
    "day_of_year",
]
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "sales_predictions_total",
    "Total number of sales predictions",
)

PREDICTED_SALES = Histogram(
    "predicted_sales_dollars",
    "Distribution of predicted restaurant sales",
    buckets=[1000, 1500, 2000, 2500, 3000, 4000, 5000],
)

class SalesPredictionRequest(BaseModel):
    prediction_date: date
    temperature: float = Field(ge=-30, le=130)
    promotion: bool


class SalesPredictionResponse(BaseModel):
    prediction_date: date
    predicted_sales: float
    model_name: str
    model_alias: str


if MODEL_URI.startswith("models:/"):
    mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(
    title="Restaurant Sales Prediction API",
    description="Predict daily restaurant sales using the champion model.",
    version="1.0.0",
)

@app.get("/")
def root() -> dict:
    return {
        "message": "Restaurant Sales Prediction API",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }

@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = perf_counter()
    response = await call_next(request)

    duration = perf_counter() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path,
    ).observe(duration)

    return response

@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
    }


@app.post("/predict", response_model=SalesPredictionResponse)
def predict_sales(
    request: SalesPredictionRequest,
) -> SalesPredictionResponse:
    prediction_date = request.prediction_date
    day_of_week = prediction_date.weekday()

    model_input = pd.DataFrame(
        [
            {
                "day_of_week": float(day_of_week),
                "is_weekend": float(day_of_week in [5, 6]),
                "temperature": float(request.temperature),
                "promotion": float(request.promotion),
                "month": float(prediction_date.month),
                "day_of_year": float(
                    prediction_date.timetuple().tm_yday
                ),
            }
        ],
        columns=FEATURES,
    )

    prediction = model.predict(model_input)
    predicted_sales = max(0.0, float(prediction[0]))

    PREDICTION_COUNT.inc()
    PREDICTED_SALES.observe(predicted_sales)

    return SalesPredictionResponse(
        prediction_date=prediction_date,
        predicted_sales=round(predicted_sales, 2),
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )
