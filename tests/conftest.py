import numpy as np
import pandas as pd
import pytest

from autoanalyst.config import Settings
from autoanalyst.ingest import load_csv
from autoanalyst.profiling import build_profile


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def make_messy_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Every problem the fallback plan has to handle, in one table."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n).round(1)
    age[rng.choice(n, 20, replace=False)] = np.nan
    age[[3, 7]] = [500.0, 480.0]
    income = [f"${v:,.0f}" for v in rng.normal(52000, 9000, n)]
    for i in rng.choice(n, 4, replace=False):
        income[i] = "unknown"
    days = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    signup = [d.strftime("%Y-%m-%d") if i % 3 else d.strftime("%m/%d/%Y") for i, d in enumerate(days)]
    city = rng.choice(["Paris", "Berlin", "Rome", "Madrid"], n).astype(object)
    city[rng.choice(n, 10, replace=False)] = None
    notes = [f"customer note number {i} with some longer free text attached" for i in range(n)]
    churn = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "income": income,
        "signup_date": signup,
        "city": city,
        "member": rng.choice(["yes", "no"], n),
        "notes": notes,
        "constant": "x",
        "churn": churn,
    })
    duplicates = df.iloc[[10, 20, 30, 40, 50]]
    return pd.concat([df, duplicates], ignore_index=True)


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    return make_messy_frame()


@pytest.fixture
def messy_loaded(messy_frame, settings):
    return load_csv(messy_frame.to_csv(index=False).encode(), "messy.csv", settings)


@pytest.fixture
def messy_profile(messy_loaded, settings):
    return build_profile(messy_loaded, target="churn", settings=settings)
