import re

import numpy as np
import pandas as pd

from autoanalyst.config import Settings, get_settings
from autoanalyst.ingest.loader import LoadedData
from autoanalyst.schemas import ColumnProfile, NumericStats, Profile
from autoanalyst.toolbox.cleaning import (
    CURRENCY_CHARS,
    is_numeric,
    parse_datetime_like,
    parse_number_like,
)

DETECT_SAMPLE = 1000
MAX_TEXT = 50


def _clip(value) -> str:
    text = str(value)
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 3] + "..."


def _round(x: float) -> float:
    return 0.0 if pd.isna(x) else float(np.format_float_positional(x, precision=6, fractional=False))


def _date_shape(value: str) -> str:
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", value.strip()))


def _looks_like_id(name: str, values: pd.Series, unique_ratio: float, settings: Settings) -> bool:
    if len(values) < 20 or unique_ratio < settings.id_unique_ratio:
        return False
    if not is_numeric(values):
        return True
    if not np.all(np.mod(values.astype(float), 1) == 0):
        return False
    lowered = name.lower()
    sequential = (values.max() - values.min() + 1) <= 1.1 * values.nunique()
    return lowered.endswith("id") or lowered.startswith("unnamed") or lowered == "index" or sequential


def _detect_kind(series: pd.Series, non_null: pd.Series, n_unique: int, unique_ratio: float,
                 settings: Settings) -> str:
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric(series):
        return "id" if _looks_like_id(str(series.name), non_null, unique_ratio, settings) else "numeric"

    sample = non_null.astype(str)
    if len(sample) > DETECT_SAMPLE:
        sample = sample.sample(DETECT_SAMPLE, random_state=0)
    if parse_number_like(sample).notna().mean() >= settings.detect_ratio:
        return "numeric_text"
    has_digit = sample.str.contains(r"\d", regex=True).mean() >= settings.detect_ratio
    if has_digit and parse_datetime_like(sample).notna().mean() >= settings.detect_ratio:
        return "datetime"
    if sample.str.len().mean() > 30 and unique_ratio > 0.5:
        return "text"
    if _looks_like_id(str(series.name), non_null, unique_ratio, settings):
        return "id"
    return "categorical"


def _numeric_stats(values: pd.Series) -> NumericStats:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return NumericStats(
        mean=_round(values.mean()),
        std=_round(values.std()),
        min=_round(values.min()),
        q1=_round(q1),
        median=_round(median),
        q3=_round(q3),
        max=_round(values.max()),
        skew=_round(values.skew()) if len(values) > 2 else 0.0,
    )


def _outlier_counts(values: pd.Series, extreme_factor: float) -> tuple[int, int]:
    if values.nunique() <= 10:
        return 0, 0
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1

    def outside(k: float) -> int:
        return int(((values < q1 - k * iqr) | (values > q3 + k * iqr)).sum())

    return outside(1.5), outside(extreme_factor)


def profile_column(series: pd.Series, settings: Settings | None = None) -> ColumnProfile:
    settings = settings or get_settings()
    n = len(series)
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / max(len(non_null), 1)
    kind = _detect_kind(series, non_null, n_unique, unique_ratio, settings)

    numbers = None
    n_unparseable = 0
    if kind in ("numeric", "id") and is_numeric(series):
        numbers = non_null.astype(float)
    elif kind == "numeric_text":
        parsed = parse_number_like(non_null)
        n_unparseable = int(parsed.isna().sum())
        numbers = parsed.dropna()
    elif kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
        n_unparseable = int(parse_datetime_like(non_null).isna().sum())

    stats, mild, extreme = None, 0, 0
    if numbers is not None and len(numbers):
        stats = _numeric_stats(numbers)
        if kind != "id":
            mild, extreme = _outlier_counts(numbers, settings.outlier_iqr_factor)

    top_values = {}
    if kind in ("categorical", "boolean", "constant", "text", "id", "datetime"):
        counts = non_null.astype(str).value_counts().head(5)
        top_values = {_clip(k): int(v) for k, v in counts.items()}

    flags = []
    missing_frac = (n - len(non_null)) / max(n, 1)
    if missing_frac >= settings.mostly_missing_frac:
        flags.append("mostly_missing")
    if kind == "constant":
        flags.append("constant")
    if kind == "id":
        flags.append("id_like")
    if kind == "categorical" and n_unique > settings.ordinal_max_levels:
        flags.append("high_cardinality")
    if stats and abs(stats.skew) > 1:
        flags.append("skewed")
    if extreme:
        flags.append("extreme_outliers")
    if kind == "numeric_text" and non_null.astype(str).str.contains(f"[{re.escape(CURRENCY_CHARS)}]").any():
        flags.append("currency_symbols")
    if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
        if non_null.astype(str).head(DETECT_SAMPLE).map(_date_shape).nunique() > 1:
            flags.append("mixed_date_formats")

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        kind=kind,
        n_missing=n - len(non_null),
        missing_frac=round(missing_frac, 4),
        n_unique=n_unique,
        unique_ratio=round(unique_ratio, 4),
        samples=[_clip(v) for v in non_null.drop_duplicates().head(3)],
        top_values=top_values,
        stats=stats,
        n_outliers_mild=mild,
        n_outliers_extreme=extreme,
        n_unparseable=n_unparseable,
        flags=flags,
    )


def guess_target(df: pd.DataFrame) -> str:
    return str(df.columns[-1])


def build_profile(
    data: LoadedData, target: str | None = None, settings: Settings | None = None
) -> Profile:
    """Compact, LLM-safe summary of the data: schema, stats and a handful of sample rows."""
    settings = settings or get_settings()
    df = data.df
    if target is not None and target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    guessed = target is None
    target = target or guess_target(df)

    columns = [profile_column(df[c], settings) for c in df.columns]
    warnings = list(data.warnings)
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        warnings.append(f"{n_duplicates} exact duplicate rows found")
    for col in columns:
        if col.kind == "datetime":
            warnings.append(
                f"Date column '{col.name}' detected: a random train/test split may be invalid "
                "if rows depend on time"
            )
        if "mixed_date_formats" in col.flags:
            warnings.append(f"Column '{col.name}' mixes date formats; day/month order may be ambiguous")

    target_profile = next(c for c in columns if c.name == target)
    if guessed:
        warnings.append(f"No target given; guessed '{target}' (last column)")
    if target_profile.n_missing:
        warnings.append(f"Target '{target}' has {target_profile.n_missing} missing values")
    if target_profile.kind in ("id", "text", "constant", "datetime"):
        warnings.append(f"Target '{target}' looks like a {target_profile.kind} column")

    sample_rows = [
        {str(k): "" if pd.isna(v) else _clip(v) for k, v in row.items()}
        for row in df.head(5).to_dict(orient="records")
    ]
    return Profile(
        file_name=data.file_name,
        data_hash=data.data_hash,
        encoding=data.encoding,
        delimiter=data.delimiter,
        n_rows=len(df),
        n_cols=df.shape[1],
        n_duplicates=n_duplicates,
        target=target,
        target_guessed=guessed,
        columns=columns,
        sample_rows=sample_rows,
        warnings=warnings,
    )
