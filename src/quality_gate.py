import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_PATH = Path("data/raw/restaurant_sales.csv")

MAX_MAE = 150.0
MAX_RMSE = 190.0
MIN_R2 = 0.80

FEATURES = [
    "day_of_week",
    "is_weekend",
    "temperature",
    "promotion",
    "month",
    "day_of_year",
]

TARGET = "daily_sales"


def main() -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear

    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    model = LinearRegression()
    model.fit(train_data[FEATURES], train_data[TARGET])

    predictions = model.predict(test_data[FEATURES])
    actual = test_data[TARGET]

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    print("Model quality gate")
    print(f"MAE:  ${mae:,.2f} — maximum allowed: ${MAX_MAE:,.2f}")
    print(f"RMSE: ${rmse:,.2f} — maximum allowed: ${MAX_RMSE:,.2f}")
    print(f"R²:   {r2:.4f} — minimum allowed: {MIN_R2:.2f}")

    failures = []

    if mae > MAX_MAE:
        failures.append(f"MAE exceeded ${MAX_MAE:,.2f}")

    if rmse > MAX_RMSE:
        failures.append(f"RMSE exceeded ${MAX_RMSE:,.2f}")

    if r2 < MIN_R2:
        failures.append(f"R² fell below {MIN_R2:.2f}")

    if failures:
        print("\nQUALITY GATE FAILED")

        for failure in failures:
            print(f"- {failure}")

        sys.exit(1)

    print("\nQUALITY GATE PASSED")


if __name__ == "__main__":
    main()