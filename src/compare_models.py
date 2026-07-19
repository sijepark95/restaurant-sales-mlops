from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_PATH = Path("data/raw/restaurant_sales.csv")

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
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear

    return data


def evaluate_model(
    name: str,
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": name,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data = prepare_data(data)

    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    x_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    x_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, model in models.items():
        result = evaluate_model(
            name,
            model,
            x_train,
            y_train,
            x_test,
            y_test,
        )
        results.append(result)

    results_table = pd.DataFrame(results)
    results_table = results_table.sort_values("mae")

    print("\nModel comparison")
    print(
        results_table.to_string(
            index=False,
            formatters={
                "mae": "${:,.2f}".format,
                "rmse": "${:,.2f}".format,
                "r2": "{:.4f}".format,
            },
        )
    )

    best_model = results_table.iloc[0]
    print(f"\nBest model by MAE: {best_model['model']}")
    print(f"Best MAE: ${best_model['mae']:,.2f}")


if __name__ == "__main__":
    main()