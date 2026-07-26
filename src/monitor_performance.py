import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PRODUCTION_PATH = Path("data/production/requests.csv")
REPORT_PATH = Path("reports/production_performance.csv")

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

MAX_PRODUCTION_MAE = 200.0


def build_model_input(data: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(data["date"])

    return pd.DataFrame(
        {
            "day_of_week": dates.dt.dayofweek.astype(float),
            "is_weekend": (
                dates.dt.dayofweek.isin([5, 6]).astype(float)
            ),
            "temperature": data["temperature"].astype(float),
            "promotion": data["promotion"].astype(float),
            "month": dates.dt.month.astype(float),
            "day_of_year": dates.dt.dayofyear.astype(float),
        },
        columns=FEATURES,
    )


def main() -> None:
    production = pd.read_csv(
        PRODUCTION_PATH,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "temperature",
        "promotion",
        "daily_sales",
    }

    missing_columns = required_columns - set(production.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if MODEL_URI.startswith("models:/"):
        mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.pyfunc.load_model(MODEL_URI)
    model_input = build_model_input(production)

    predictions = model.predict(model_input)

    results = production.copy()
    results["predicted_sales"] = predictions
    results["absolute_error"] = (
        results["daily_sales"] - results["predicted_sales"]
    ).abs()

    actual = results["daily_sales"]
    predicted = results["predicted_sales"]

    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(
        actual,
        predicted,
    ) ** 0.5
    r2 = r2_score(actual, predicted)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)

    print("Production model performance")
    print("-" * 48)
    print(f"Records: {len(results)}")
    print(f"MAE:     ${mae:,.2f}")
    print(f"RMSE:    ${rmse:,.2f}")
    print(f"R²:       {r2:.4f}")
    print(f"Report:   {REPORT_PATH}")
    print("-" * 48)

    if mae > MAX_PRODUCTION_MAE:
        print(
            f"PERFORMANCE ALERT: MAE ${mae:,.2f} exceeds "
            f"${MAX_PRODUCTION_MAE:,.2f}"
        )
        sys.exit(1)

    print("MODEL PERFORMANCE ACCEPTABLE")


if __name__ == "__main__":
    main()