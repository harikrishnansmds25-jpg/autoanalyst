import numpy as np
import pandas as pd
import pytest

from autoanalyst.agents import fallback_plan
from autoanalyst.schemas import Plan, PlanStep
from autoanalyst.toolbox.executor import execute
from autoanalyst.validators import check_plan, validate_execution
from autoanalyst.validators.data_checks import (
    check_distribution_shift,
    check_dtypes,
    check_no_duplicates,
    check_no_leakage,
    check_no_unexpected_nulls,
    check_row_count,
    check_target_untouched,
)


def _step(action, column=None, **params):
    return PlanStep(column=column, action=action, reason="test", params=params)


def _rules(results):
    return {r.name for r in results}



def test_row_count():
    before = pd.DataFrame({"a": range(100)})
    assert check_row_count(before, before.iloc[:95], 0.1).passed
    assert not check_row_count(before, before.iloc[:80], 0.1).passed
    assert not check_row_count(before.iloc[:5], before, 0.1).passed


def test_no_unexpected_nulls():
    clean = pd.DataFrame({"a": [1.0, 2.0]})
    dirty = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
    assert check_no_unexpected_nulls({"train": clean}).passed
    result = check_no_unexpected_nulls({"train": clean, "holdout": dirty})
    assert not result.passed and "a" in result.message


def test_dtypes():
    assert check_dtypes({"t": pd.DataFrame({"a": [1.0], "b": [True]})}).passed
    text = check_dtypes({"t": pd.DataFrame({"a": ["x"]})})
    assert not text.passed and text.details["non_numeric"] == ["a"]
    inf = check_dtypes({"t": pd.DataFrame({"a": [np.inf]})})
    assert not inf.passed and inf.details["infinite"] == ["a"]


def test_target_untouched():
    before = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, np.nan]})
    assert check_target_untouched(before, before.iloc[[0, 2]], "y").passed
    changed = before.copy()
    changed.loc[1, "y"] = 5
    assert not check_target_untouched(before, changed, "y").passed
    recast = before.astype({"y": "str"})
    assert not check_target_untouched(before, recast, "y").passed
    assert not check_target_untouched(before, before.drop(columns="y"), "y").passed
    assert check_target_untouched(before, before, None).severity == "info"


def test_no_duplicates():
    before = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    assert check_no_duplicates(before, before.iloc[[0, 2]]).passed
    assert not check_no_duplicates(before, before).passed


def test_distribution_shift_numeric():
    rng = np.random.default_rng(0)
    observed = pd.Series(rng.normal(0, 1, 1000), name="v")
    light = observed.copy()
    light[:100] = np.nan
    heavy = observed.copy()
    heavy[:800] = np.nan
    args = (0.25, 0.6, 1.5, 0.2)
    assert check_distribution_shift(light, light.fillna(light.median()), *args).passed
    bad = check_distribution_shift(heavy, heavy.fillna(heavy.median()), *args)
    assert not bad.passed and bad.details["std_ratio"] < 0.6


def test_distribution_shift_categorical():
    before = pd.Series(["a"] * 50 + ["b"] * 30 + [None] * 20, name="c")
    ok = check_distribution_shift(before, before.fillna("a"), 0.25, 0.6, 1.5, 0.2)
    assert ok.passed
    skewed = pd.Series(["a"] * 10 + ["b"] * 10 + [None] * 80, name="c")
    bad = check_distribution_shift(skewed, skewed.fillna("a"), 0.25, 0.6, 1.5, 0.2)
    assert not bad.passed


def test_leakage_detection():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 200))
    clean = pd.DataFrame({"x": rng.normal(size=200), "g": rng.choice(["a", "b", "c"], 200)})
    assert check_no_leakage(clean, y, ["drop_column"], 0.98).passed

    copy = clean.assign(leak=y.astype(str))
    assert "copy of the target" in check_no_leakage(copy, y, [], 0.98).message

    near = clean.assign(near=y + rng.normal(0, 0.01, 200))
    assert not check_no_leakage(near, y, [], 0.98).passed

    fitted = check_no_leakage(clean, y, ["impute_median"], 0.98)
    assert not fitted.passed and "full data" in fitted.message

    mapping = clean.assign(code=np.where(y == 1, "yes", "no"))
    suspect = check_no_leakage(mapping.drop(columns="g"), y.map({0: "n", 1: "p"}), [], 0.98)
    assert not suspect.passed



def test_fallback_plan_passes_plan_checks(messy_profile, settings):
    assert check_plan(fallback_plan(messy_profile, settings), messy_profile, settings) == []


@pytest.mark.parametrize("steps, rule", [
    ([_step("keep", "nope")], "plan.unknown_column"),
    ([_step("impute_median", "churn")], "plan.target_touched"),
    ([_step("drop_column", "age"), _step("impute_median", "age")], "plan.conflict"),
    ([_step("impute_median", "city"), _step("encode_onehot", "city")], "plan.bad_action"),
    ([_step("impute_mode", "city")], "plan.not_encoded"),
    ([_step("encode_onehot", "city")], "plan.not_imputed"),
    ([_step("impute_median", "age"), _step("impute_mode", "age")], "plan.two_imputers"),
    ([_step("coerce_numeric", "city")], "plan.bad_action"),
    ([_step("parse_dates", "age")], "plan.bad_action"),
    ([_step("cap_outliers", "age", factor=0.1)], "plan.bad_param"),
    ([_step("impute_median", "age", strategy="mean")], "plan.bad_param"),
])
def test_plan_errors(messy_profile, settings, steps, rule):
    base = [s for s in fallback_plan(messy_profile, settings).steps
            if s.column not in {st.column for st in steps}]
    plan = Plan(target="churn", steps=base + steps)
    errors = [r for r in check_plan(plan, messy_profile, settings) if r.severity == "error"]
    assert rule in _rules(errors)


def test_plan_warnings(messy_profile, settings):
    plan = Plan(target="churn", steps=[
        s for s in fallback_plan(messy_profile, settings).steps
        if s.action != "drop_duplicates" and s.column != "customer_id"
    ])
    rules = _rules(check_plan(plan, messy_profile, settings))
    assert {"plan.duplicates_kept", "plan.kept_suspect"} <= rules


def test_plan_dropping_everything(messy_profile, settings):
    plan = Plan(target="churn", steps=[
        _step("drop_column", c.name) for c in messy_profile.columns if c.name != "churn"
    ])
    assert "plan.no_features" in _rules(check_plan(plan, messy_profile, settings))


def test_plan_step_schema_rejects_bad_shapes():
    with pytest.raises(ValueError):
        PlanStep(action="drop_duplicates", column="a", reason="x" * 5)
    with pytest.raises(ValueError):
        PlanStep(action="impute_median", reason="needs column")
    with pytest.raises(ValueError):
        PlanStep(action="run_python", column="a", reason="not allowed")



def test_fallback_plan_passes_all_validators(messy_loaded, messy_profile, settings):
    plan = fallback_plan(messy_profile, settings)
    report = validate_execution(messy_loaded.df, plan, execute(messy_loaded.df, plan), settings)
    assert report.passed, [c.message for c in report.errors]
    assert {"row_count", "no_duplicates", "target_untouched", "no_nulls", "dtypes", "no_leakage"} <= {
        c.name for c in report.checks
    }


def test_validators_catch_a_bad_plan(messy_loaded, messy_profile, settings):
    plan = Plan(target="churn", steps=[
        _step("drop_column", "notes"), _step("drop_column", "constant"),
        _step("drop_column", "signup_date"), _step("drop_column", "income"),
        _step("encode_onehot", "city"), _step("encode_ordinal", "member"),
    ])
    report = validate_execution(messy_loaded.df, plan, execute(messy_loaded.df, plan), settings)
    failed = {c.name for c in report.errors}
    assert {"no_nulls", "no_duplicates"} <= failed


def test_dry_run_failure_is_reported_not_raised(messy_loaded, settings):
    plan = Plan(target="churn", steps=[_step("impute_median", "city")])
    report = validate_execution(messy_loaded.df, plan, execute(messy_loaded.df, plan), settings)
    assert "dry_run" in {c.name for c in report.errors}
