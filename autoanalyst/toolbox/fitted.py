"""Tools that learn from data (medians, categories, bounds).

They are only ever fitted on a training split, inside a ColumnTransformer. The single-column
functions below take an explicit `fit_df` so the same code can be tested for leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from autoanalyst.schemas import PlanStep
from autoanalyst.toolbox.cleaning import ToolError, is_numeric, require_column

FITTED_ORDER = {
    "impute_median": 0,
    "impute_mode": 0,
    "cap_outliers": 1,
    "encode_onehot": 2,
    "encode_ordinal": 2,
    "scale_standard": 3,
}
NUMERIC_ONLY = {"impute_median", "cap_outliers", "scale_standard"}
ENCODERS = {"encode_onehot", "encode_ordinal"}


class OutlierCapper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip values to [q1 - factor*IQR, q3 + factor*IQR], with quartiles learned in fit."""

    def __init__(self, factor: float = 3.0):
        self.factor = factor

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def _as_strings(X):
    frame = pd.DataFrame(X).astype(object)
    missing = frame.isna()
    return frame.astype(str).astype(object).mask(missing, np.nan)


def make_transformer(action: str, params: dict | None = None):
    params = params or {}
    if action == "impute_median":
        return SimpleImputer(strategy="median", keep_empty_features=True)
    if action == "impute_mode":
        return SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    if action == "cap_outliers":
        return OutlierCapper(factor=float(params.get("factor", 3.0)))
    if action == "encode_onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    if action == "encode_ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        )
    if action == "scale_standard":
        return StandardScaler()
    raise ToolError(f"'{action}' is not a fitted tool")


def build_column_pipeline(
    steps: list[PlanStep], categorical: bool, scale_numeric: bool = False
) -> Pipeline | None:
    actions = [s for s in sorted(steps, key=lambda s: FITTED_ORDER[s.action])]
    names = {s.action for s in actions}
    pipe = []
    if categorical and actions:
        pipe.append(("as_str", FunctionTransformer(_as_strings, feature_names_out="one-to-one")))
    for i, step in enumerate(actions):
        pipe.append((f"{i}_{step.action}", make_transformer(step.action, step.params)))
    wants_scale = scale_numeric and "scale_standard" not in names and "encode_onehot" not in names
    if wants_scale and (not categorical or "encode_ordinal" in names):
        pipe.append(("auto_scale", StandardScaler()))
    return Pipeline(pipe) if pipe else None


def build_preprocessor(
    X: pd.DataFrame,
    steps_by_column: dict[str, list[PlanStep]],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """One small pipeline per feature column; columns without fitted steps pass through."""
    transformers = []
    for i, column in enumerate(X.columns):
        pipe = build_column_pipeline(
            steps_by_column.get(column, []),
            categorical=not is_numeric(X[column]),
            scale_numeric=scale_numeric,
        )
        if pipe is not None:
            transformers.append((f"c{i}", pipe, [column]))
    ct = ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def _fit_apply(
    action: str, df: pd.DataFrame, column: str, fit_df: pd.DataFrame | None, **params
) -> pd.DataFrame:
    fit_df = df if fit_df is None else fit_df
    require_column(df, column)
    require_column(fit_df, column)
    if action in NUMERIC_ONLY and not is_numeric(fit_df[column]):
        raise ToolError(f"{action} needs a numeric column; '{column}' is {fit_df[column].dtype}")
    step = PlanStep(column=column, action=action, reason="direct call", params=params)
    ct = build_preprocessor(fit_df[[column]], {column: [step]})
    ct.fit(fit_df[[column]])
    transformed = ct.transform(df[[column]])
    pos = df.columns.get_loc(column)
    return pd.concat([df.iloc[:, :pos], transformed, df.iloc[:, pos + 1 :]], axis=1)


def impute_median(df: pd.DataFrame, column: str, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _fit_apply("impute_median", df, column, fit_df)


def impute_mode(df: pd.DataFrame, column: str, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _fit_apply("impute_mode", df, column, fit_df)


def cap_outliers(
    df: pd.DataFrame, column: str, factor: float = 3.0, fit_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    return _fit_apply("cap_outliers", df, column, fit_df, factor=factor)


def encode_onehot(df: pd.DataFrame, column: str, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _fit_apply("encode_onehot", df, column, fit_df)


def encode_ordinal(df: pd.DataFrame, column: str, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _fit_apply("encode_ordinal", df, column, fit_df)


def scale_standard(df: pd.DataFrame, column: str, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _fit_apply("scale_standard", df, column, fit_df)
