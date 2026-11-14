"""The complete set of actions a plan may use. Nothing outside this registry can be executed."""

from autoanalyst.toolbox import cleaning, fitted

CLEAN_TOOLS = {
    "coerce_numeric": cleaning.coerce_numeric,
    "parse_dates": cleaning.parse_dates,
    "drop_duplicates": cleaning.drop_duplicates,
    "drop_column": cleaning.drop_column,
}
# Type fixes first so duplicates are compared on normalised values, and before
# identifier columns are dropped so distinct records are not merged.
CLEAN_STAGE = {"coerce_numeric": 0, "parse_dates": 0, "drop_duplicates": 1, "drop_column": 2}

FITTED_TOOLS = {
    "impute_median": fitted.impute_median,
    "impute_mode": fitted.impute_mode,
    "cap_outliers": fitted.cap_outliers,
    "encode_onehot": fitted.encode_onehot,
    "encode_ordinal": fitted.encode_ordinal,
    "scale_standard": fitted.scale_standard,
}

ALLOWED_PARAMS = {
    "parse_dates": {"dayfirst": bool, "expand": bool},
    "cap_outliers": {"factor": (int, float)},
}

ALL_ACTIONS = {"keep"} | set(CLEAN_TOOLS) | set(FITTED_TOOLS)
