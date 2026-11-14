import numpy as np
import pandas as pd
from pydantic import BaseModel

from autoanalyst.config import Settings, get_settings
from autoanalyst.schemas import Task
from autoanalyst.toolbox.cleaning import is_numeric
from autoanalyst.training.target import infer_task

MAX_CORR_COLUMNS = 30


class NumericSummary(BaseModel):
    column: str
    mean: float
    std: float
    min: float
    median: float
    max: float
    skew: float
    n_missing: int


class CategoricalSummary(BaseModel):
    column: str
    n_levels: int
    top: str
    top_share: float
    n_missing: int


class TargetSummary(BaseModel):
    name: str
    task: Task
    n_missing: int
    class_counts: dict[str, int] = {}
    minority_share: float | None = None
    stats: NumericSummary | None = None


class Association(BaseModel):
    feature: str
    measure: str
    value: float


class EdaResult(BaseModel):
    n_rows: int
    n_features: int
    numeric: list[NumericSummary]
    categorical: list[CategoricalSummary]
    correlations: dict[str, dict[str, float]]
    strong_pairs: list[tuple[str, str, float]]
    target: TargetSummary
    associations: list[Association]
    warnings: list[str]


def _r(x: float) -> float:
    return 0.0 if pd.isna(x) else round(float(x), 4)


def numeric_summary(s: pd.Series) -> NumericSummary:
    v = s.dropna().astype(float)
    return NumericSummary(
        column=str(s.name), mean=_r(v.mean()), std=_r(v.std()), min=_r(v.min()),
        median=_r(v.median()), max=_r(v.max()), skew=_r(v.skew()), n_missing=int(s.isna().sum()),
    )


def categorical_summary(s: pd.Series) -> CategoricalSummary:
    counts = s.dropna().astype(str).value_counts()
    return CategoricalSummary(
        column=str(s.name), n_levels=len(counts),
        top=str(counts.index[0]) if len(counts) else "",
        top_share=_r(counts.iloc[0] / counts.sum()) if len(counts) else 0.0,
        n_missing=int(s.isna().sum()),
    )


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: share of the numeric variance explained by the categorical grouping (0..1)."""
    frame = pd.DataFrame({"c": categories.astype(str), "v": values.astype(float)}).dropna()
    if frame["v"].var() == 0 or len(frame) < 3:
        return 0.0
    grand = frame["v"].mean()
    groups = frame.groupby("c")["v"]
    between = (groups.count() * (groups.mean() - grand) ** 2).sum()
    total = ((frame["v"] - grand) ** 2).sum()
    return float(np.sqrt(between / total))


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a.astype(str), b.astype(str))
    if min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    expected = observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / n
    chi2 = ((observed - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / n / (min(table.shape) - 1)))


def _association(x: pd.Series, y: pd.Series, task: Task) -> Association:
    feature = str(x.name)
    if is_numeric(x) and task == "regression":
        return Association(feature=feature, measure="pearson", value=_r(x.corr(y.astype(float))))
    if is_numeric(x):
        return Association(feature=feature, measure="eta", value=_r(correlation_ratio(y, x)))
    if task == "regression":
        return Association(feature=feature, measure="eta", value=_r(correlation_ratio(x, y)))
    return Association(feature=feature, measure="cramers_v", value=_r(cramers_v(x, y)))


def run_eda(df: pd.DataFrame, target: str, task: Task | None = None,
            settings: Settings | None = None) -> EdaResult:
    """Descriptive analysis of the cleaned data. Nothing here feeds into model fitting."""
    settings = settings or get_settings()
    y = df[target]
    task = task or infer_task(y)
    X = df.drop(columns=[target])
    numeric_cols = [c for c in X.columns if is_numeric(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    warnings = []

    corr_cols = numeric_cols[:MAX_CORR_COLUMNS]
    corr = X[corr_cols].corr() if len(corr_cols) > 1 else pd.DataFrame()
    strong = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= settings.strong_corr:
                strong.append((a, b, _r(r)))
                warnings.append(f"'{a}' and '{b}' are highly correlated (r = {r:.2f})")

    numeric = [numeric_summary(X[c]) for c in numeric_cols]
    skewed = [n.column for n in numeric if abs(n.skew) > 1]
    if skewed:
        warnings.append(f"Skewed numeric columns: {', '.join(skewed)}")

    target_summary = TargetSummary(name=target, task=task, n_missing=int(y.isna().sum()))
    if target_summary.n_missing:
        warnings.append(f"{target_summary.n_missing} rows have no target value and will be excluded")
    if task == "classification":
        counts = y.dropna().astype(str).value_counts()
        target_summary.class_counts = {k: int(v) for k, v in counts.items()}
        target_summary.minority_share = _r(counts.min() / counts.sum())
        if target_summary.minority_share < settings.minority_share:
            warnings.append(
                f"Class imbalance: smallest class '{counts.index[-1]}' is "
                f"{target_summary.minority_share:.1%} of rows"
            )
    elif is_numeric(y):
        target_summary.stats = numeric_summary(y)
        if abs(target_summary.stats.skew) > 1:
            warnings.append(f"Target '{target}' is skewed (skew {target_summary.stats.skew:.2f})")

    known = y.notna()
    associations = [_association(X.loc[known, c], y[known], task) for c in X.columns]
    associations.sort(key=lambda a: abs(a.value), reverse=True)
    if len(df) < 100:
        warnings.append(f"Only {len(df)} rows; results will be unstable")

    return EdaResult(
        n_rows=len(df),
        n_features=X.shape[1],
        numeric=numeric,
        categorical=[categorical_summary(X[c]) for c in categorical_cols],
        correlations={a: {b: _r(corr.loc[a, b]) for b in corr.columns} for a in corr.columns},
        strong_pairs=strong,
        target=target_summary,
        associations=associations,
        warnings=warnings,
    )
