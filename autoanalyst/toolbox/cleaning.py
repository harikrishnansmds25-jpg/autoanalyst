"""Stateless cleaning tools. Safe to run on the full dataset: none of them learn from the data."""

import re
import warnings

import numpy as np
import pandas as pd

CURRENCY_CHARS = "$€£¥₹"
_NUMBER_JUNK = re.compile(rf"[{re.escape(CURRENCY_CHARS)}\s,]")
DATE_PARTS = {"year": "year", "month": "month", "day": "day", "dayofweek": "day_of_week"}


class ToolError(ValueError):
    pass


def require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ToolError(f"Column '{column}' not found")


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def parse_number_like(series: pd.Series) -> pd.Series:
    """Parse strings such as '$1,200', '15%', '(300)' into floats. Unparseable values become NaN."""
    if is_numeric(series):
        return series.astype(float)
    text = series.astype("string").str.strip()
    negative = text.str.match(r"^\(.*\)$", na=False)
    text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    text = text.str.replace(_NUMBER_JUNK, "", regex=True).str.rstrip("%")
    parsed = pd.to_numeric(text, errors="coerce")
    values = parsed.to_numpy(dtype=float, na_value=np.nan)
    flip = negative.to_numpy(dtype=bool, na_value=False)
    values[flip] = -values[flip]
    return pd.Series(values, index=series.index, name=series.name)


def parse_datetime_like(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=dayfirst)


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    require_column(df, column)
    return df.drop(columns=[column])


def drop_duplicates(df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first. The original index is preserved."""
    return df.loc[~df.duplicated(keep="first")].copy()


def coerce_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    require_column(df, column)
    out = df.copy()
    out[column] = parse_number_like(df[column])
    return out


def parse_dates(
    df: pd.DataFrame, column: str, expand: bool = True, dayfirst: bool = False
) -> pd.DataFrame:
    """Parse a date column. With expand=True it is replaced by numeric year/month/day/dayofweek columns."""
    require_column(df, column)
    if is_numeric(df[column]):
        raise ToolError(f"Column '{column}' is numeric, not a date")
    out = df.copy()
    parsed = parse_datetime_like(df[column], dayfirst=dayfirst)
    if not expand:
        out[column] = parsed
        return out
    position = out.columns.get_loc(column)
    out = out.drop(columns=[column])
    for offset, (part, attr) in enumerate(DATE_PARTS.items()):
        values = getattr(parsed.dt, attr).astype(float)
        out.insert(position + offset, f"{column}_{part}", values)
    return out


def date_feature_names(column: str) -> list[str]:
    return [f"{column}_{part}" for part in DATE_PARTS]
