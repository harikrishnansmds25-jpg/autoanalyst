"""Deterministic corrupted copies of the reference datasets.

Each corruption is one a real upload might contain. Two traps are added on purpose:
an identifier column and a column that leaks the target. Validators cannot always see a leak
(a categorical code that maps 1:1 onto a class is only a warning), so these measure silent mistakes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parents[1] / "tests" / "data"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def add_nulls(df: pd.DataFrame, column: str, frac: float, rng) -> None:
    idx = rng.choice(len(df), int(frac * len(df)), replace=False)
    df.loc[df.index[idx], column] = np.nan


def add_duplicates(df: pd.DataFrame, frac: float, rng) -> pd.DataFrame:
    idx = rng.choice(len(df), max(1, int(frac * len(df))), replace=False)
    return pd.concat([df, df.iloc[idx]], ignore_index=True)


def add_outliers(df: pd.DataFrame, column: str, n: int, factor: float, rng) -> None:
    idx = rng.choice(len(df), n, replace=False)
    df.loc[df.index[idx], column] = df[column].iloc[idx] * factor


def as_text_with_junk(df: pd.DataFrame, column: str, frac: float, rng) -> None:
    text = df[column].map(lambda v: "" if pd.isna(v) else f"{v:g}").astype(object)
    idx = rng.choice(len(df), int(frac * len(df)), replace=False)
    text.iloc[idx] = "unknown"
    df[column] = text.replace("", np.nan)


def as_currency(df: pd.DataFrame, column: str, symbol: str = "$") -> None:
    df[column] = df[column].map(lambda v: np.nan if pd.isna(v) else f"{symbol}{v:,.2f}")


def add_mixed_dates(df: pd.DataFrame, column: str, rng) -> None:
    days = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, len(df)), unit="D")
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"]
    df[column] = [d.strftime(formats[i % 3]) for i, d in enumerate(days)]


def add_id(df: pd.DataFrame, column: str) -> None:
    df.insert(0, column, np.arange(1000, 1000 + len(df)))


def iris() -> tuple[pd.DataFrame, str]:
    return pd.read_csv(DATA / "iris.csv"), "species"


def titanic() -> tuple[pd.DataFrame, str]:
    return pd.read_csv(DATA / "titanic.csv"), "Survived"


def california() -> tuple[pd.DataFrame, str]:
    return pd.read_csv(DATA / "california_housing.csv"), "MedHouseVal"


def iris_corrupt(seed: int = 1) -> tuple[pd.DataFrame, str]:
    rng = _rng(seed)
    df, target = iris()
    add_nulls(df, "sepal length (cm)", 0.1, rng)
    add_nulls(df, "petal width (cm)", 0.1, rng)
    add_outliers(df, "sepal width (cm)", 3, 20, rng)
    as_text_with_junk(df, "petal length (cm)", 0.05, rng)
    add_mixed_dates(df, "measured_on", rng)
    add_id(df, "record_id")
    df["species_code"] = df[target].map({"setosa": "S1", "versicolor": "V2", "virginica": "V3"})
    df = add_duplicates(df, 0.04, rng)
    return df, target


def titanic_corrupt(seed: int = 2) -> tuple[pd.DataFrame, str]:
    rng = _rng(seed)
    df, target = titanic()
    as_currency(df, "Fare")
    as_text_with_junk(df, "Age", 0.03, rng)
    add_nulls(df, "Sex", 0.05, rng)
    add_mixed_dates(df, "booking_date", rng)
    df["outcome_label"] = df[target].map({0: "lost", 1: "saved"})
    df = add_duplicates(df, 0.03, rng)
    return df, target


def california_corrupt(seed: int = 3) -> tuple[pd.DataFrame, str]:
    rng = _rng(seed)
    df, target = california()
    df = df.sample(5000, random_state=seed).reset_index(drop=True)
    as_currency(df, "MedInc")
    as_text_with_junk(df, "HouseAge", 0.04, rng)
    add_nulls(df, "AveRooms", 0.15, rng)
    add_outliers(df, "Population", 10, 50, rng)
    add_mixed_dates(df, "survey_date", rng)
    add_id(df, "record_id")
    df["value_index"] = (df[target] * 100).round(0)
    df = add_duplicates(df, 0.02, rng)
    return df, target


DATASETS = {
    "iris": iris,
    "titanic": titanic,
    "california": california,
    "iris_corrupt": iris_corrupt,
    "titanic_corrupt": titanic_corrupt,
    "california_corrupt": california_corrupt,
}


def load(name: str) -> tuple[bytes, str]:
    """CSV bytes and target, as the system would receive them from an upload."""
    df, target = DATASETS[name]()
    return df.to_csv(index=False).encode(), target
