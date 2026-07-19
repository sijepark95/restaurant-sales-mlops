from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_DAYS = 730


def generate_sales_data() -> pd.DataFrame:
    """Generate two years of simulated daily restaurant sales data."""

    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2024-01-01", periods=NUMBER_OF_DAYS)

    data = pd.DataFrame({"date": dates})
    data["day_of_week"] = data["date"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["temperature"] = (
        58
        + 22 * np.sin(2 * np.pi * data["date"].dt.dayofyear / 365)
        + rng.normal(0, 5, NUMBER_OF_DAYS)
    ).round(1)

    data["promotion"] = rng.binomial(1, 0.15, NUMBER_OF_DAYS)
    data["online_orders"] = (
        45
        + data["is_weekend"] * 20
        + data["promotion"] * 15
        + rng.normal(0, 8, NUMBER_OF_DAYS)
    ).round().clip(lower=0).astype(int)

    data["daily_sales"] = (
        1200
        + data["is_weekend"] * 500
        + data["promotion"] * 300
        + data["online_orders"] * 12
        + data["temperature"] * 3
        + rng.normal(0, 150, NUMBER_OF_DAYS)
    ).round(2)

    return data


def main() -> None:
    output_path = Path("data/raw/restaurant_sales.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_sales_data()
    data.to_csv(output_path, index=False)

    print(f"Created {len(data)} rows")
    print(f"Saved dataset to: {output_path}")
    print(data.head())


if __name__ == "__main__":
    main()