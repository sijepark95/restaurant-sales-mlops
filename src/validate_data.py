from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/restaurant_sales.csv")

REQUIRED_COLUMNS = {
    "date",
    "day_of_week",
    "is_weekend",
    "temperature",
    "promotion",
    "online_orders",
    "daily_sales",
}


def validate_data(data: pd.DataFrame) -> None:
    """Validate the schema and quality of the sales dataset."""

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Dataset is empty")

    if data.isna().any().any():
        null_counts = data.isna().sum()
        raise ValueError(f"Missing values found:\n{null_counts[null_counts > 0]}")

    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    if not data["day_of_week"].between(0, 6).all():
        raise ValueError("day_of_week must be between 0 and 6")

    if not data["is_weekend"].isin([0, 1]).all():
        raise ValueError("is_weekend must contain only 0 or 1")

    if not data["promotion"].isin([0, 1]).all():
        raise ValueError("promotion must contain only 0 or 1")

    if (data["online_orders"] < 0).any():
        raise ValueError("online_orders cannot be negative")

    if (data["daily_sales"] <= 0).any():
        raise ValueError("daily_sales must be greater than zero")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    validate_data(data)

    print("Data validation passed")
    print(f"Rows: {len(data)}")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Average daily sales: ${data['daily_sales'].mean():,.2f}")
    print(f"Missing values: {data.isna().sum().sum()}")
    print(f"Duplicate dates: {data['date'].duplicated().sum()}")


if __name__ == "__main__":
    main()