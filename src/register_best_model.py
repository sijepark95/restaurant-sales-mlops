import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "restaurant-sales-prediction"
REGISTERED_MODEL_NAME = "restaurant-sales-forecaster"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_type = 'linear-regression'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError("No completed experiment runs found")

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    model_type = best_run["params.model_type"]
    mae = best_run["metrics.mae"]
    r2 = best_run["metrics.r2"]

    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="validation_status",
        value="approved",
    )

    print("Best model registered")
    print(f"Model type: {model_type}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Registered name: {REGISTERED_MODEL_NAME}")
    print(f"Version: {registered_model.version}")
    print("Alias: champion")


if __name__ == "__main__":
    main()