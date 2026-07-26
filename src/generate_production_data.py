from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_PATH = Path("data/raw/restaurant_sales.csv")
OUTPUT_PATH = Path("data/production/requests.csv")
RANDOM_SEED = 42


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    reference = pd.read_csv(
        REFERENCE_PATH,
        parse_dates=["date"],
    )

    production = reference.sample(
        n=200,
        replace=True,
        random_state=RANDOM_SEED,
    ).copy()

    production["date"] = production["date"] + pd.DateOffset(years=2)

    # Simulate warmer-than-usual production traffic.
    production["temperature"] += 12

    # Simulate an increase in promotional days.
    production["promotion"] = rng.binomial(
        n=1,
        p=0.40,
        size=len(production),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    production.to_csv(OUTPUT_PATH, index=False)

    print(f"Created {len(production)} production records")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()