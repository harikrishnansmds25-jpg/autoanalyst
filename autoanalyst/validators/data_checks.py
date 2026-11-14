"""Deterministic checks on executed data. Their verdict cannot be overridden by any agent."""

import numpy as np
import pandas as pd

from autoanalyst.schemas import CheckResult
from autoanalyst.toolbox.cleaning import is_numeric
from autoanalyst.toolbox.registry import CLEAN_TOOLS


def check_row_count(before: pd.DataFrame, after: pd.DataFrame, max_drop_frac: float) -> CheckResult:
    dropped = len(before) - len(after)
    if dropped < 0:
        return CheckResult(name="row_count", passed=False, message=f"{-dropped} rows were added")
    frac = dropped / max(len(before), 1)
    return CheckResult(
        name="row_count",
        passed=frac <= max_drop_frac,
        message=f"{dropped} of {len(before)} rows removed ({frac:.1%}, limit {max_drop_frac:.0%})",
        details={"rows_before": len(before), "rows_after": len(after)},
    )


def check_no_unexpected_nulls(frames: dict[str, pd.DataFrame]) -> CheckResult:
    found = {name: sorted(f.columns[f.isna().any()].astype(str)) for name, f in frames.items()}
    found = {k: v for k, v in found.items() if v}
    if not found:
        return CheckResult(name="no_nulls", passed=True, message="No missing values after preprocessing")
    cols = sorted({c for v in found.values() for c in v})
    return CheckResult(
        name="no_nulls", passed=False,
        message=f"Missing values remain in: {', '.join(cols)}", details=found,
    )


def check_dtypes(frames: dict[str, pd.DataFrame]) -> CheckResult:
    bad, infinite = set(), set()
    for frame in frames.values():
        for col in frame.columns:
            if not (is_numeric(frame[col]) or pd.api.types.is_bool_dtype(frame[col])):
                bad.add(str(col))
            elif is_numeric(frame[col]) and np.isinf(frame[col].to_numpy(dtype=float)).any():
                infinite.add(str(col))
    problems = []
    if bad:
        problems.append(f"non-numeric model inputs: {', '.join(sorted(bad))}")
    if infinite:
        problems.append(f"infinite values in: {', '.join(sorted(infinite))}")
    return CheckResult(
        name="dtypes",
        passed=not problems,
        message="; ".join(problems) or "All model inputs are numeric and finite",
        details={"non_numeric": sorted(bad), "infinite": sorted(infinite)},
    )


def check_target_untouched(before: pd.DataFrame, after: pd.DataFrame, target: str | None) -> CheckResult:
    if target is None:
        return CheckResult(name="target_untouched", passed=True, severity="info",
                           message="No target selected yet")
    if target not in after.columns:
        return CheckResult(name="target_untouched", passed=False,
                           message=f"Target '{target}' was removed")
    original = before.loc[after.index, target]
    if not original.equals(after[target]):
        return CheckResult(name="target_untouched", passed=False,
                           message=f"Target '{target}' values or dtype were changed")
    return CheckResult(name="target_untouched", passed=True, message=f"Target '{target}' unchanged")


def check_no_duplicates(before: pd.DataFrame, after: pd.DataFrame) -> CheckResult:
    """Retained records must be unique as originally recorded (index is preserved by the executor)."""
    retained = before.loc[after.index]
    n = int(retained.duplicated().sum())
    return CheckResult(
        name="no_duplicates", passed=n == 0,
        message=f"{n} duplicate rows remain" if n else "No duplicate rows remain",
    )


def _tvd(before: pd.Series, after: pd.Series) -> float:
    p = before.value_counts(normalize=True)
    q = after.value_counts(normalize=True)
    return float(0.5 * p.subtract(q, fill_value=0).abs().sum())


def check_distribution_shift(
    before: pd.Series,
    after: pd.Series,
    max_mean_shift_std: float,
    min_std_ratio: float,
    max_std_ratio: float,
    max_tvd: float,
) -> CheckResult:
    """Compare a column's observed values with the same column after imputation/capping."""
    name = f"shift.{before.name}"
    observed = before.dropna()
    if len(observed) < 2:
        return CheckResult(name=name, passed=True, severity="info", message="Too few values to compare")

    if is_numeric(observed) and is_numeric(after):
        b, a = observed.astype(float), after.astype(float)
        std = b.std()
        if std == 0:
            return CheckResult(name=name, passed=True, severity="info", message="Constant column")
        shift = abs(a.mean() - b.mean()) / std
        ratio = a.std() / std
        passed = shift <= max_mean_shift_std and min_std_ratio <= ratio <= max_std_ratio
        return CheckResult(
            name=name, passed=passed,
            message=f"'{before.name}': mean shift {shift:.2f} std, std ratio {ratio:.2f}",
            details={"mean_shift_std": round(shift, 4), "std_ratio": round(ratio, 4)},
        )

    tvd = _tvd(observed.astype(str), after.astype(str))
    return CheckResult(
        name=name, passed=tvd <= max_tvd,
        message=f"'{before.name}': category distribution moved by {tvd:.2f} (limit {max_tvd})",
        details={"tvd": round(tvd, 4)},
    )


def check_no_leakage(
    X: pd.DataFrame, y: pd.Series | None, applied_actions: list[str], corr_limit: float
) -> CheckResult:
    """Hard failures: fitted steps run on full data, target copies, near-perfect correlation.
    Categorical features that map one-to-one onto the target are reported as a warning."""
    errors, suspects = [], []
    fitted_on_full = sorted(set(applied_actions) - set(CLEAN_TOOLS))
    if fitted_on_full:
        errors.append(f"fitted steps ran on the full data: {', '.join(fitted_on_full)}")

    if y is not None:
        y_known = y.dropna()
        for col in X.columns:
            x = X.loc[y_known.index, col]
            if x.astype(str).equals(y_known.astype(str)):
                errors.append(f"'{col}' is a copy of the target")
                continue
            if is_numeric(x) and is_numeric(y_known) and x.nunique() > 1 and y_known.nunique() > 1:
                corr = x.corr(y_known)
                if pd.notna(corr) and abs(corr) >= corr_limit:
                    errors.append(f"'{col}' correlates {corr:.3f} with the target")
            elif not is_numeric(x) and 2 <= x.nunique() <= len(x) / 5 and y_known.nunique() > 1:
                per_level = pd.DataFrame({"x": x, "y": y_known}).dropna().groupby("x")["y"].nunique()
                if per_level.max() == 1:
                    suspects.append(f"'{col}' determines the target exactly")

    if errors:
        return CheckResult(name="no_leakage", passed=False, message="; ".join(errors),
                           details={"errors": errors, "suspects": suspects})
    if suspects:
        return CheckResult(name="no_leakage", passed=False, severity="warning",
                           message="; ".join(suspects), details={"suspects": suspects})
    return CheckResult(name="no_leakage", passed=True, message="No leakage signals found")
