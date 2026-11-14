import pandas as pd
import pytest

from autoanalyst.schemas import Plan, PlanStep
from autoanalyst.toolbox.executor import ExecutionError, dry_run, execute


def _step(action, column=None, **params):
    return PlanStep(column=column, action=action, reason="test", params=params)


@pytest.fixture
def df():
    return pd.DataFrame({
        "id": range(8),
        "price": ["$1", "$1", "$3", "$4", "$5", "$6", "$7", "$8"],
        "when": ["2021-01-01"] * 2 + ["2021-02-01"] * 6,
        "cat": ["a", "a", "b", None, "a", "b", "a", "b"],
        "y": [0, 0, 1, 0, 1, 0, 1, 1],
    })


def test_clean_steps_run_in_stage_order(df):
    plan = Plan(target="y", steps=[
        _step("drop_column", "id"),
        _step("drop_duplicates"),
        _step("coerce_numeric", "price"),
    ])
    result = execute(df, plan)
    assert [e["action"] for e in result.log] == ["coerce_numeric", "drop_duplicates", "drop_column"]
    # duplicates are judged before the id column is dropped, so no rows are lost
    assert len(result.df) == 8


def test_duplicates_judged_on_normalised_values():
    df = pd.DataFrame({"p": ["$1,000", "1000", "5"], "y": [1, 1, 0]})
    plan = Plan(target="y", steps=[_step("coerce_numeric", "p"), _step("drop_duplicates")])
    result = execute(df, plan)
    assert list(result.df.index) == [0, 2]


def test_fitted_steps_are_deferred_not_applied(df):
    plan = Plan(target="y", steps=[_step("impute_mode", "cat"), _step("encode_onehot", "cat")])
    result = execute(df, plan)
    assert result.df["cat"].isna().sum() == 1
    assert [s.action for s in result.deferred] == ["impute_mode", "encode_onehot"]
    assert result.log == []


def test_date_steps_follow_derived_columns(df):
    plan = Plan(target="y", steps=[_step("parse_dates", "when"), _step("impute_median", "when")])
    result = execute(df, plan)
    assert result.column_map["when"] == ["when_year", "when_month", "when_day", "when_dayofweek"]
    assert set(result.steps_by_feature()) == set(result.column_map["when"])


def test_execution_error_names_the_step(df):
    plan = Plan(target="y", steps=[_step("parse_dates", "id")])
    with pytest.raises(ExecutionError, match="parse_dates"):
        execute(df, plan)


def test_dry_run_fits_on_train_only(df):
    plan = Plan(target="y", steps=[
        _step("drop_column", "id"),
        _step("coerce_numeric", "price"),
        _step("parse_dates", "when"),
        _step("impute_mode", "cat"),
        _step("encode_onehot", "cat"),
    ])
    result = execute(df, plan)
    dry = dry_run(result, "y", seed=0, holdout_frac=0.25)
    assert "y" not in dry.train_after.columns
    assert len(dry.train_after) == 6 and len(dry.holdout_after) == 2
    assert dry.train_after.notna().all().all()
    assert {"cat_a", "cat_b"} <= set(dry.train_after.columns)
