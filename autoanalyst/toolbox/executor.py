from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from autoanalyst.schemas import Plan, PlanStep
from autoanalyst.toolbox.cleaning import date_feature_names
from autoanalyst.toolbox.fitted import build_preprocessor
from autoanalyst.toolbox.registry import CLEAN_STAGE, CLEAN_TOOLS, FITTED_TOOLS


class ExecutionError(RuntimeError):
    def __init__(self, step: PlanStep, cause: Exception):
        self.step = step
        self.cause = cause
        super().__init__(f"{step.action}({step.column}) failed: {cause}")


@dataclass
class ExecutionResult:
    df: pd.DataFrame
    log: list[dict] = field(default_factory=list)
    deferred: list[PlanStep] = field(default_factory=list)
    column_map: dict[str, list[str]] = field(default_factory=dict)

    def steps_by_feature(self) -> dict[str, list[PlanStep]]:
        """Fitted steps keyed by the current column names (date columns map to their parts)."""
        out: dict[str, list[PlanStep]] = {}
        for step in self.deferred:
            for name in self.column_map.get(step.column, []):
                out.setdefault(name, []).append(step)
        return out


def execute(df: pd.DataFrame, plan: Plan) -> ExecutionResult:
    """Apply the stateless cleaning steps to the full data. Fitted steps are only recorded."""
    column_map = {c: [c] for c in df.columns}
    clean = [s for s in plan.steps if s.action in CLEAN_TOOLS]
    clean.sort(key=lambda s: CLEAN_STAGE[s.action])
    deferred = [s for s in plan.steps if s.action in FITTED_TOOLS]

    current = df
    log = []
    for step in clean:
        rows_before = len(current)
        try:
            current = CLEAN_TOOLS[step.action](current, step.column, **step.params)
        except Exception as exc:
            raise ExecutionError(step, exc) from exc
        if step.action == "drop_column":
            column_map[step.column] = []
        elif step.action == "parse_dates" and step.params.get("expand", True):
            column_map[step.column] = date_feature_names(step.column)
        log.append({
            "action": step.action,
            "column": step.column,
            "rows_before": rows_before,
            "rows_after": len(current),
        })
    return ExecutionResult(df=current, log=log, deferred=deferred, column_map=column_map)


@dataclass
class DryRun:
    train_before: pd.DataFrame
    holdout_before: pd.DataFrame
    train_after: pd.DataFrame
    holdout_after: pd.DataFrame


def dry_run(
    result: ExecutionResult, target: str | None, seed: int = 42, holdout_frac: float = 0.25
) -> DryRun:
    """Fit the deferred steps on a train split and apply them to a holdout, as training will."""
    X = result.df.drop(columns=[target]) if target in result.df.columns else result.df
    train, holdout = train_test_split(X, test_size=holdout_frac, random_state=seed)
    pre = build_preprocessor(train, result.steps_by_feature())
    pre.fit(train)
    return DryRun(train, holdout, pre.transform(train), pre.transform(holdout))
