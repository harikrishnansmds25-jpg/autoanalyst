"""Convert between a Plan and the editable table shown at checkpoint 1."""

import pandas as pd
from pydantic import ValidationError

from autoanalyst.schemas import Plan, PlanStep

ALL_ROWS = "(all rows)"


def plan_frame(plan: Plan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"column": s.column or ALL_ROWS, "action": s.action, "reason": s.reason,
             "factor": s.params.get("factor"), "dayfirst": s.params.get("dayfirst")}
            for s in plan.steps
        ],
        columns=["column", "action", "reason", "factor", "dayfirst"],
    )


def frame_plan(frame: pd.DataFrame, target: str) -> tuple[Plan | None, str | None]:
    """Edited table back to a Plan. Returns (plan, None) or (None, error message)."""
    steps = []
    try:
        for row in frame.to_dict("records"):
            if not row.get("action") or pd.isna(row.get("action")):
                continue
            params = {}
            if row.get("factor") is not None and pd.notna(row.get("factor")):
                params["factor"] = float(row["factor"])
            if row.get("dayfirst") is not None and pd.notna(row.get("dayfirst")):
                params["dayfirst"] = bool(row["dayfirst"])
            column = row.get("column")
            column = None if column is None or pd.isna(column) or column == ALL_ROWS else column
            reason = row.get("reason")
            reason = reason if isinstance(reason, str) and len(reason.strip()) >= 3 else "Edited by the user"
            steps.append(PlanStep(column=column, action=row["action"], reason=reason, params=params))
    except ValidationError as exc:
        first = exc.errors()[0]
        return None, f"row {len(steps) + 1}: {first['msg']}"
    return Plan(steps=steps, target=target, source="user"), None
