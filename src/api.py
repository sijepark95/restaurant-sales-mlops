from datetime import date
import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


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


class SalesPredictionRequest(BaseModel):
    prediction_date: date
    temperature: float = Field(ge=-30, le=130)
    promotion: bool


class SalesPredictionResponse(BaseModel):
    prediction_date: date
    predicted_sales: float
    model_name: str
    model_alias: str


mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(
    title="Restaurant Sales Prediction API",
    description="Predict daily restaurant sales using the champion model.",
    version="1.0.0",
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

    return SalesPredictionResponse(
        prediction_date=prediction_date,
        predicted_sales=round(predicted_sales, 2),
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )