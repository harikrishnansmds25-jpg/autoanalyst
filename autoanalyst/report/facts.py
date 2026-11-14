"""facts.json: the only input the report writer sees. Every number in the narrative must come from here."""

import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from pydantic import BaseModel

from autoanalyst.eda.analysis import EdaResult
from autoanalyst.schemas import CheckResult, DecisionEntry, Plan, Profile, ValidationReport
from autoanalyst.training.trainer import TrainResult

PACKAGES = ["pandas", "numpy", "scikit-learn", "pydantic", "matplotlib", "python-docx", "jinja2"]


class DatasetFacts(BaseModel):
    file_name: str
    data_hash: str
    n_rows: int
    n_cols: int
    n_duplicates: int
    target: str
    task: str
    n_rows_clean: int
    n_features_clean: int


class ColumnIssue(BaseModel):
    column: str
    kind: str
    n_missing: int
    missing_frac: float
    flags: list[str]


class Summary(BaseModel):
    n_plan_steps: int
    n_dropped_columns: int
    n_columns_with_missing: int
    n_checks: int
    n_checks_passed: int
    n_warnings: int


class Facts(BaseModel):
    dataset: DatasetFacts
    column_issues: list[ColumnIssue]
    summary: Summary
    plan: Plan
    decision_log: list[DecisionEntry]
    validation: list[CheckResult]
    eda: EdaResult
    model: TrainResult
    warnings: list[str]
    run: dict[str, Any]


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_facts(
    profile: Profile,
    plan: Plan,
    validation: ValidationReport,
    n_rows_clean: int,
    eda: EdaResult,
    model: TrainResult,
    decision_log: list[DecisionEntry] | None = None,
    run: dict[str, Any] | None = None,
) -> Facts:
    issues = [
        ColumnIssue(column=c.name, kind=c.kind, n_missing=c.n_missing,
                    missing_frac=c.missing_frac, flags=c.flags)
        for c in profile.columns if c.n_missing or c.flags or c.kind in ("id", "text", "constant")
    ]
    warnings = _unique(
        profile.warnings
        + [c.message for c in validation.checks if not c.passed]
        + eda.warnings
        + model.warnings
    )
    summary = Summary(
        n_plan_steps=len(plan.steps),
        n_dropped_columns=sum(s.action == "drop_column" for s in plan.steps),
        n_columns_with_missing=sum(c.n_missing > 0 for c in profile.columns),
        n_checks=len(validation.checks),
        n_checks_passed=sum(c.passed for c in validation.checks),
        n_warnings=len(warnings),
    )
    run_info = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "seed": model.seed,
        "libraries": library_versions(),
    }
    run_info.update(run or {})
    return Facts(
        dataset=DatasetFacts(
            file_name=profile.file_name, data_hash=profile.data_hash, n_rows=profile.n_rows,
            n_cols=profile.n_cols, n_duplicates=profile.n_duplicates, target=model.target,
            task=model.task, n_rows_clean=n_rows_clean, n_features_clean=len(model.features),
        ),
        column_issues=issues,
        summary=summary,
        plan=plan,
        decision_log=decision_log or [],
        validation=validation.checks,
        eda=eda,
        model=model,
        warnings=warnings,
        run=run_info,
    )
