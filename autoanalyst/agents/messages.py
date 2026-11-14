"""Schemas for everything an agent may say. Replies that do not validate are rejected and retried."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from autoanalyst.schemas import TABLE_ACTIONS, Action, Plan, PlanStep


class StepOut(BaseModel):
    # Required (though nullable): constrained decoders drop optional fields.
    column: str | None
    action: Action
    reason: str = Field(min_length=3)
    factor: float | None = None
    dayfirst: bool | None = None

    @model_validator(mode="after")
    def _column_matches_action(self) -> "StepOut":
        if self.action in TABLE_ACTIONS and self.column is not None:
            raise ValueError(f"{self.action} is a table-level action; set column to null")
        if self.action not in TABLE_ACTIONS and not self.column:
            raise ValueError(f"{self.action} needs a column")
        return self

    def to_step(self) -> PlanStep:
        params = {}
        if self.factor is not None:
            params["factor"] = self.factor
        if self.dayfirst is not None:
            params["dayfirst"] = self.dayfirst
        return PlanStep(column=self.column, action=self.action, reason=self.reason, params=params)


class PlannerOutput(BaseModel):
    steps: list[StepOut] = Field(min_length=1)
    notes: str = ""

    def to_plan(self, target: str | None) -> Plan:
        return Plan(steps=[s.to_step() for s in self.steps], target=target, source="planner")


def plan_to_output(plan: Plan, notes: str = "") -> PlannerOutput:
    return PlannerOutput(
        steps=[
            StepOut(column=s.column, action=s.action, reason=s.reason,
                    factor=s.params.get("factor"), dayfirst=s.params.get("dayfirst"))
            for s in plan.steps
        ],
        notes=notes,
    )


class CriticIssue(BaseModel):
    column: str | None
    problem: str = Field(min_length=3)
    suggestion: str = Field(min_length=3)
    severity: Literal["blocker", "minor"]


class CriticOutput(BaseModel):
    verdict: Literal["approve", "revise"]
    issues: list[CriticIssue] = []
    summary: str = Field(min_length=3)

    @model_validator(mode="after")
    def _consistent(self) -> "CriticOutput":
        if self.verdict == "approve" and any(i.severity == "blocker" for i in self.issues):
            raise ValueError("verdict is approve but a blocker issue is listed")
        if self.verdict == "revise" and not self.issues:
            raise ValueError("verdict is revise but no issue is listed")
        return self
