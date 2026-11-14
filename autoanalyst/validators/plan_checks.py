"""Static checks on a plan against the profile, run before anything is executed."""

from autoanalyst.config import Settings, get_settings
from autoanalyst.schemas import CheckResult, ColumnProfile, Plan, Profile
from autoanalyst.toolbox.fitted import ENCODERS, NUMERIC_ONLY
from autoanalyst.toolbox.registry import ALLOWED_PARAMS

IMPUTERS = {"impute_median", "impute_mode"}


def _effective_numeric(col: ColumnProfile, actions: list[str]) -> bool:
    if col.kind in ("numeric", "id"):
        return True
    if col.kind == "numeric_text":
        return "coerce_numeric" in actions
    if col.kind == "datetime":
        return "parse_dates" in actions
    return False


def check_plan(plan: Plan, profile: Profile, settings: Settings | None = None) -> list[CheckResult]:
    settings = settings or get_settings()
    issues: list[CheckResult] = []

    def issue(rule: str, message: str, severity: str = "error", column: str | None = None) -> None:
        issues.append(CheckResult(
            name=f"plan.{rule}", passed=False, severity=severity, message=message,
            details={"column": column} if column else {},
        ))

    known = {c.name for c in profile.columns}
    for step in plan.steps:
        if step.column is not None and step.column not in known:
            issue("unknown_column", f"Column '{step.column}' does not exist", column=step.column)
        allowed = ALLOWED_PARAMS.get(step.action, {})
        for key, value in step.params.items():
            if key not in allowed:
                issue("bad_param", f"{step.action} does not take parameter '{key}'", column=step.column)
            elif not isinstance(value, allowed[key]):
                issue("bad_param", f"{step.action}.{key} has the wrong type", column=step.column)
        if step.action == "cap_outliers" and not 1.0 <= float(step.params.get("factor", 3.0)) <= 10.0:
            issue("bad_param", "cap_outliers factor must be between 1 and 10", column=step.column)

    target = plan.target or profile.target
    kept_features = 0
    for col in profile.columns:
        actions = [s.action for s in plan.steps_for(col.name)]
        name = col.name
        if name == target:
            if any(a != "keep" for a in actions):
                issue("target_touched", f"Target '{name}' must not be transformed by the plan", column=name)
            continue
        if "drop_column" in actions:
            if set(actions) - {"drop_column", "keep"}:
                issue("conflict", f"'{name}' is dropped but also has other steps", column=name)
            continue
        kept_features += 1

        if len(actions) != len(set(actions)):
            issue("repeated_step", f"'{name}' has a repeated step", "warning", name)
        if sum(a in IMPUTERS for a in set(actions)) > 1:
            issue("two_imputers", f"'{name}' has more than one imputation step", column=name)
        if sum(a in ENCODERS for a in set(actions)) > 1:
            issue("two_encoders", f"'{name}' has more than one encoding step", column=name)
        if "coerce_numeric" in actions and col.kind not in ("numeric_text", "numeric"):
            issue("bad_action", f"coerce_numeric on '{name}' ({col.kind}) would turn it into nulls", column=name)
        if "parse_dates" in actions and col.kind != "datetime":
            issue("bad_action", f"parse_dates on '{name}' which is not a date column ({col.kind})", column=name)

        numeric = _effective_numeric(col, actions)
        for action in set(actions) & NUMERIC_ONLY:
            if not numeric:
                issue("bad_action", f"{action} needs a numeric column; '{name}' is {col.kind}", column=name)
        if not numeric and not set(actions) & ENCODERS:
            issue("not_encoded", f"'{name}' ({col.kind}) is kept but never encoded", column=name)
        if col.n_missing + col.n_unparseable > 0 and not set(actions) & IMPUTERS:
            issue("not_imputed", f"'{name}' has missing values but no imputation step", column=name)
        if "encode_onehot" in actions and col.n_unique > settings.onehot_max_levels * 2:
            issue("wide_onehot", f"One-hot on '{name}' creates {col.n_unique} columns", "warning", name)
        if col.kind in ("id", "constant", "text") or "mostly_missing" in col.flags:
            issue("kept_suspect", f"'{name}' is kept although it looks {col.kind} or mostly missing",
                  "warning", name)
        if col.kind == "datetime" and "parse_dates" not in actions:
            issue("date_as_category", f"Date column '{name}' is not parsed", "warning", name)

    if profile.n_duplicates and not any(s.action == "drop_duplicates" for s in plan.steps):
        issue("duplicates_kept", f"{profile.n_duplicates} duplicate rows are not removed", "warning")
    if kept_features == 0:
        issue("no_features", "The plan drops every feature column")
    return issues
