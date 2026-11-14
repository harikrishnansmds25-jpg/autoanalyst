import pandas as pd

from autoanalyst.config import Settings, get_settings
from autoanalyst.schemas import CheckResult, Plan, ValidationReport
from autoanalyst.toolbox.cleaning import is_numeric
from autoanalyst.toolbox.executor import ExecutionResult, dry_run
from autoanalyst.toolbox.fitted import build_column_pipeline
from autoanalyst.validators.data_checks import (
    check_distribution_shift,
    check_dtypes,
    check_no_duplicates,
    check_no_leakage,
    check_no_unexpected_nulls,
    check_row_count,
    check_target_untouched,
)

IMPUTERS = {"impute_median", "impute_mode"}


def _fit_column(train: pd.DataFrame, feature: str, steps) -> pd.Series:
    pipe = build_column_pipeline(steps, categorical=not is_numeric(train[feature]))
    if pipe is None:
        return train[feature]
    pipe.set_output(transform="pandas")
    return pipe.fit_transform(train[[feature]]).iloc[:, 0]


def _shift_checks(train: pd.DataFrame, result: ExecutionResult, settings: Settings) -> list[CheckResult]:
    """Measure what imputation alone does to a column. Capping is applied to both sides,
    because changing the tails is its purpose."""
    checks = []
    for feature, steps in result.steps_by_feature().items():
        if feature not in train.columns or not any(s.action in IMPUTERS for s in steps):
            continue
        capping = [s for s in steps if s.action == "cap_outliers"]
        reference = _fit_column(train, feature, capping).rename(feature)
        after = _fit_column(train, feature, [s for s in steps if s.action in IMPUTERS | {"cap_outliers"}])
        checks.append(check_distribution_shift(
            reference, after,
            settings.max_mean_shift_std, settings.min_std_ratio,
            settings.max_std_ratio, settings.max_category_tvd,
        ))
    return checks


def validate_execution(
    raw: pd.DataFrame, plan: Plan, result: ExecutionResult, settings: Settings | None = None
) -> ValidationReport:
    """Run every data validator on an executed plan, including a train/holdout dry run."""
    settings = settings or get_settings()
    target = plan.target
    after = result.df
    checks = [
        check_row_count(raw, after, settings.max_row_drop_frac),
        check_no_duplicates(raw, after),
        check_target_untouched(raw, after, target),
    ]

    try:
        dry = dry_run(result, target, settings.seed, settings.holdout_frac)
    except Exception as exc:
        checks.append(CheckResult(name="dry_run", passed=False,
                                  message=f"Preprocessing failed on a train split: {exc}"))
    else:
        frames = {"train": dry.train_after, "holdout": dry.holdout_after}
        checks.append(check_no_unexpected_nulls(frames))
        checks.append(check_dtypes(frames))
        try:
            checks += _shift_checks(dry.train_before, result, settings)
        except Exception as exc:
            checks.append(CheckResult(name="shift", passed=False,
                                      message=f"Could not compare distributions: {exc}"))

    features = after.drop(columns=[target]) if target in after.columns else after
    y = after[target] if target in after.columns else None
    checks.append(check_no_leakage(features, y, [e["action"] for e in result.log], settings.leakage_corr))
    return ValidationReport(checks=checks)
