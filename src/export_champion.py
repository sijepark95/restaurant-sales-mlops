import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "restaurant-sales-forecaster"
MODEL_ALIAS = "champion"
OUTPUT_PATH = Path("models/champion")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_version.source,
    )

    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)

    shutil.copytree(downloaded_path, OUTPUT_PATH)

    print("Champion model exported")
    print(f"Model version: {model_version.version}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()