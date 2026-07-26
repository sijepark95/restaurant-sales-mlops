import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


REFERENCE_PATH = Path("data/raw/restaurant_sales.csv")
PRODUCTION_PATH = Path("data/production/requests.csv")

CONTINUOUS_FEATURES = ["temperature"]
CATEGORICAL_FEATURES = [
    "day_of_week",
    "is_weekend",
    "promotion",
]

MAX_KS_STATISTIC = 0.20
MAX_P_VALUE = 0.05
MIN_CRAMERS_V = 0.10


def calculate_cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    chi2, p_value, _, _ = chi2_contingency(table)

    total = table.to_numpy().sum()
    rows, columns = table.shape
    denominator = total * min(rows - 1, columns - 1)

    if denominator == 0:
        return 0.0, p_value

    cramers_v = np.sqrt(chi2 / denominator)
    return cramers_v, p_value


def main() -> None:
    reference = pd.read_csv(REFERENCE_PATH)
    production = pd.read_csv(PRODUCTION_PATH)

    drifted_features = []

    print("Data drift report")
    print("-" * 80)

    for feature in CONTINUOUS_FEATURES:
        statistic, p_value = ks_2samp(
            reference[feature].dropna(),
            production[feature].dropna(),
        )

        drift_detected = (
            statistic >= MAX_KS_STATISTIC
            and p_value <= MAX_P_VALUE
        )

        status = "DRIFT" if drift_detected else "OK"

        print(
            f"{feature:15} test=KS         "
            f"effect={statistic:.4f} "
            f"p={p_value:.6f} status={status}"
        )

        if drift_detected:
            drifted_features.append(feature)

    for feature in CATEGORICAL_FEATURES:
        categories = sorted(
            set(reference[feature].dropna())
            | set(production[feature].dropna())
        )

        table = pd.DataFrame(
            {
                "reference": (
                    reference[feature]
                    .value_counts()
                    .reindex(categories, fill_value=0)
                ),
                "production": (
                    production[feature]
                    .value_counts()
                    .reindex(categories, fill_value=0)
                ),
            }
        ).T

        effect_size, p_value = calculate_cramers_v(table)

        drift_detected = (
            effect_size >= MIN_CRAMERS_V
            and p_value <= MAX_P_VALUE
        )

        status = "DRIFT" if drift_detected else "OK"

        print(
            f"{feature:15} test=Chi-square "
            f"effect={effect_size:.4f} "
            f"p={p_value:.6f} status={status}"
        )

        if drift_detected:
            drifted_features.append(feature)

    print("-" * 80)

    if drifted_features:
        print("DRIFT DETECTED: " + ", ".join(drifted_features))
        sys.exit(10)

    print("NO SIGNIFICANT DRIFT DETECTED")


if __name__ == "__main__":
    main()