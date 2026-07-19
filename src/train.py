import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_PATH = Path("data/raw/restaurant_sales.csv")
MODEL_PATH = Path("models/random_forest.joblib")
METRICS_PATH = Path("models/metrics.json")

FEATURES = [
    "day_of_week",
    "is_weekend",
    "temperature",
    "promotion",
    "month",
    "day_of_year",
]

TARGET = "daily_sales"


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the raw dataset."""

    prepared = data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["month"] = prepared["date"].dt.month
    prepared["day_of_year"] = prepared["date"].dt.dayofyear

    return prepared


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically without shuffling."""

    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    return train_data, test_data


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data = prepare_data(data)

    train_data, test_data = split_data(data)

    x_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    x_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "r2": float(r2_score(y_test, predictions)),
        "training_rows": len(train_data),
        "test_rows": len(test_data),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("Model training completed")
    print(f"Training rows: {metrics['training_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"MAE: ${metrics['mae']:,.2f}")
    print(f"RMSE: ${metrics['rmse']:,.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()