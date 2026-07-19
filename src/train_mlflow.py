from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
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

    data[FEATURES] = data[FEATURES].astype("float64")

    return data


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("restaurant-sales-prediction")

    data = prepare_data(pd.read_csv(DATA_PATH))

    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    x_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    x_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    models = {
        "mean-baseline": DummyRegressor(strategy="mean"),
        "linear-regression": LinearRegression(),
        "random-forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            metrics = {
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(
                    mean_squared_error(y_test, predictions)
                ),
                "r2": r2_score(y_test, predictions),
            }

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("training_rows", len(train_data))
            mlflow.log_param("test_rows", len(test_data))
            mlflow.log_param("features", ",".join(FEATURES))

            if model_name == "random-forest":
                mlflow.log_param("n_estimators", 200)
                mlflow.log_param("max_depth", 10)

            mlflow.log_metrics(metrics)

            signature = infer_signature(x_train, predictions)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=x_train.head(),
            )

            print(
                f"{model_name}: "
                f"MAE=${metrics['mae']:,.2f}, "
                f"RMSE=${metrics['rmse']:,.2f}, "
                f"R²={metrics['r2']:.4f}"
            )


if __name__ == "__main__":
    main()