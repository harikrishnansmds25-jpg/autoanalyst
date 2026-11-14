import numpy as np
import pandas as pd
import pytest

from autoanalyst.toolbox import (
    ToolError,
    cap_outliers,
    coerce_numeric,
    drop_column,
    drop_duplicates,
    encode_onehot,
    encode_ordinal,
    impute_median,
    impute_mode,
    parse_dates,
    scale_standard,
)
from autoanalyst.toolbox.cleaning import parse_number_like
from autoanalyst.toolbox.registry import ALL_ACTIONS, CLEAN_TOOLS, FITTED_TOOLS
from autoanalyst.schemas import Action
from typing import get_args


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 100.0],
        "cat": ["a", "b", None, "a", "c"],
        "y": [0, 1, 0, 1, 0],
    })


def test_registry_covers_schema_actions():
    assert ALL_ACTIONS == set(get_args(Action))
    assert not set(CLEAN_TOOLS) & set(FITTED_TOOLS)


def test_drop_column(frame):
    out = drop_column(frame, "cat")
    assert list(out.columns) == ["num", "y"]
    assert "cat" in frame.columns
    with pytest.raises(ToolError):
        drop_column(frame, "missing")


def test_drop_duplicates_keeps_first_and_index():
    df = pd.DataFrame({"a": [1, 2, 1, 3, 2]}, index=[10, 11, 12, 13, 14])
    out = drop_duplicates(df)
    assert list(out.index) == [10, 11, 13]


def test_parse_number_like():
    raw = pd.Series(["$1,200", "€3.5", "15%", "(300)", " 42 ", "unknown", None])
    out = parse_number_like(raw)
    assert out.iloc[:5].tolist() == [1200.0, 3.5, 15.0, -300.0, 42.0]
    assert out.iloc[5:].isna().all()


def test_coerce_numeric():
    df = pd.DataFrame({"price": ["$10", "$2,000", "n/a?"]})
    out = coerce_numeric(df, "price")
    assert pd.api.types.is_float_dtype(out["price"])
    assert out["price"].iloc[1] == 2000.0
    assert np.isnan(out["price"].iloc[2])
    assert df["price"].iloc[0] == "$10"


def test_parse_dates_expand_and_mixed_formats():
    df = pd.DataFrame({"d": ["2021-03-04", "03/05/2021", "not a date"], "v": [1, 2, 3]})
    out = parse_dates(df, "d")
    assert list(out.columns) == ["d_year", "d_month", "d_day", "d_dayofweek", "v"]
    assert out["d_year"].iloc[:2].tolist() == [2021.0, 2021.0]
    assert out["d_month"].iloc[1] == 3.0
    assert np.isnan(out["d_year"].iloc[2])


def test_parse_dates_dayfirst_and_no_expand():
    df = pd.DataFrame({"d": ["03/05/2021"]})
    assert parse_dates(df, "d", dayfirst=True)["d_month"].iloc[0] == 5.0
    kept = parse_dates(df, "d", expand=False)
    assert pd.api.types.is_datetime64_any_dtype(kept["d"])


def test_parse_dates_rejects_numeric(frame):
    with pytest.raises(ToolError):
        parse_dates(frame, "num")


def test_impute_median_uses_fit_data_only(frame):
    out = impute_median(frame, "num")
    assert out["num"].iloc[2] == 3.0
    train = pd.DataFrame({"num": [10.0, 20.0, 30.0], "cat": ["a"] * 3, "y": [0] * 3})
    out = impute_median(frame, "num", fit_df=train)
    assert out["num"].iloc[2] == 20.0
    assert list(out.columns) == ["num", "cat", "y"]


def test_impute_median_rejects_text(frame):
    with pytest.raises(ToolError, match="numeric"):
        impute_median(frame, "cat")


def test_impute_mode(frame):
    out = impute_mode(frame, "cat")
    assert out["cat"].iloc[2] == "a"
    assert out["cat"].notna().all()


def test_cap_outliers(frame):
    filled = impute_median(frame, "num")
    out = cap_outliers(filled, "num", factor=1.5)
    q1, q3 = filled["num"].quantile([0.25, 0.75])
    assert out["num"].max() == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert out["num"].iloc[0] == 1.0


def test_cap_outliers_bounds_from_fit_data():
    train = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"v": [50.0, -50.0, np.nan]})
    out = cap_outliers(test, "v", factor=1.0, fit_df=train)
    assert out["v"].iloc[0] == pytest.approx(3.25 + 1.5)
    assert out["v"].iloc[1] == pytest.approx(1.75 - 1.5)
    assert np.isnan(out["v"].iloc[2])


def test_encode_onehot_ignores_unknown():
    train = pd.DataFrame({"c": ["a", "b", "a"]})
    test = pd.DataFrame({"c": ["b", "z"]})
    out = encode_onehot(test, "c", fit_df=train)
    assert list(out.columns) == ["c_a", "c_b"]
    assert out.iloc[0].tolist() == [0.0, 1.0]
    assert out.iloc[1].tolist() == [0.0, 0.0]


def test_encode_ordinal_unknown_is_minus_one():
    train = pd.DataFrame({"c": ["low", "high", "mid"]})
    test = pd.DataFrame({"c": ["mid", "new", None]})
    out = encode_ordinal(test, "c", fit_df=train)
    assert out["c"].tolist() == [2.0, -1.0, -1.0]


def test_encode_handles_mixed_types():
    df = pd.DataFrame({"c": pd.Series(["a", 1, 2.5, "a"], dtype=object)})
    assert encode_onehot(df, "c").shape[1] == 3


def test_scale_standard_uses_fit_stats():
    train = pd.DataFrame({"v": [0.0, 2.0, 4.0]})
    out = scale_standard(pd.DataFrame({"v": [2.0, 6.0]}), "v", fit_df=train)
    assert out["v"].iloc[0] == pytest.approx(0.0)
    assert out["v"].iloc[1] == pytest.approx(4.0 / np.std([0, 2, 4]))
    with pytest.raises(ToolError):
        scale_standard(pd.DataFrame({"v": ["a", "b"]}), "v")


def test_tools_do_not_mutate_input(frame):
    snapshot = frame.copy()
    for tool in (impute_median, cap_outliers, scale_standard):
        tool(frame, "num")
    for tool in (impute_mode, encode_onehot, encode_ordinal):
        tool(frame, "cat")
    drop_column(frame, "num")
    coerce_numeric(frame, "num")
    pd.testing.assert_frame_equal(frame, snapshot)
