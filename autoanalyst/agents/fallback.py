"""Deterministic default plan, used when the LLM is unavailable or keeps producing invalid output."""

from autoanalyst.config import Settings, get_settings
from autoanalyst.schemas import ColumnProfile, Plan, PlanStep, Profile


def _column_steps(col: ColumnProfile, settings: Settings) -> list[PlanStep]:
    name = col.name

    def step(action: str, reason: str, **params) -> PlanStep:
        return PlanStep(column=name, action=action, reason=reason, params=params)

    if col.kind == "constant":
        return [step("drop_column", "Constant column carries no information")]
    if col.missing_frac >= settings.mostly_missing_frac:
        return [step("drop_column", f"{col.missing_frac:.0%} of values are missing")]
    if col.kind == "id":
        return [step("drop_column", f"Identifier-like ({col.unique_ratio:.0%} unique values)")]
    if col.kind == "text":
        return [step("drop_column", "Free text; text modelling is out of scope")]

    missing = col.n_missing + col.n_unparseable > 0
    if col.kind == "datetime":
        steps = [step("parse_dates", "Date column split into year, month, day and weekday")]
        if missing:
            steps.append(step("impute_median", "Fill missing or unparseable dates with the median"))
        return steps

    if col.kind in ("numeric", "numeric_text"):
        steps = []
        if col.kind == "numeric_text":
            steps.append(step("coerce_numeric", "Numbers stored as text (symbols or separators)"))
        if missing:
            steps.append(step("impute_median", f"{col.n_missing + col.n_unparseable} missing or unparseable values"))
        if col.n_outliers_extreme:
            steps.append(step(
                "cap_outliers", f"{col.n_outliers_extreme} extreme outliers beyond 3 IQR",
                factor=settings.outlier_iqr_factor,
            ))
        return steps or [step("keep", "Numeric, no issues found")]

    steps = []
    if missing:
        steps.append(step("impute_mode", f"{col.n_missing} missing values filled with the most frequent value"))
    if col.n_unique <= 2:
        steps.append(step("encode_ordinal", "Binary category"))
    elif col.n_unique <= settings.onehot_max_levels:
        steps.append(step("encode_onehot", f"{col.n_unique} categories"))
    elif col.n_unique <= settings.ordinal_max_levels:
        steps.append(step("encode_ordinal", f"{col.n_unique} categories, too many for one-hot"))
    else:
        return [step("drop_column", f"{col.n_unique} categories, too many to encode")]
    return steps


def fallback_plan(profile: Profile, settings: Settings | None = None) -> Plan:
    settings = settings or get_settings()
    steps = []
    if profile.n_duplicates:
        steps.append(PlanStep(action="drop_duplicates", reason=f"{profile.n_duplicates} exact duplicate rows"))
    for col in profile.columns:
        if col.name == profile.target:
            steps.append(PlanStep(column=col.name, action="keep", reason="Target column, left untouched"))
        else:
            steps.extend(_column_steps(col, settings))
    return Plan(steps=steps, target=profile.target, source="fallback")
