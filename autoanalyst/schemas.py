from typing import Literal

from pydantic import BaseModel, Field, model_validator

ColumnKind = Literal[
    "numeric", "boolean", "categorical", "datetime", "numeric_text", "text", "id", "constant"
]

Action = Literal[
    "keep",
    "drop_column",
    "drop_duplicates",
    "parse_dates",
    "coerce_numeric",
    "impute_median",
    "impute_mode",
    "cap_outliers",
    "encode_onehot",
    "encode_ordinal",
    "scale_standard",
]

TABLE_ACTIONS = {"drop_duplicates"}

Task = Literal["classification", "regression"]


class NumericStats(BaseModel):
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skew: float


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: ColumnKind
    n_missing: int
    missing_frac: float
    n_unique: int
    unique_ratio: float
    samples: list[str] = []
    top_values: dict[str, int] = {}
    stats: NumericStats | None = None
    n_outliers_mild: int = 0
    n_outliers_extreme: int = 0
    n_unparseable: int = 0
    flags: list[str] = []


class Profile(BaseModel):
    file_name: str
    data_hash: str
    encoding: str
    delimiter: str
    n_rows: int
    n_cols: int
    n_duplicates: int
    target: str | None = None
    target_guessed: bool = False
    columns: list[ColumnProfile]
    sample_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


class PlanStep(BaseModel):
    column: str | None = None
    action: Action
    reason: str = Field(min_length=3)
    params: dict[str, float | int | str | bool] = {}

    @model_validator(mode="after")
    def _column_matches_action(self) -> "PlanStep":
        if self.action in TABLE_ACTIONS and self.column is not None:
            raise ValueError(f"{self.action} is a table-level action and takes no column")
        if self.action not in TABLE_ACTIONS and not self.column:
            raise ValueError(f"{self.action} needs a column")
        return self


class Plan(BaseModel):
    steps: list[PlanStep]
    target: str | None = None
    source: Literal["fallback", "planner", "user"] = "fallback"

    def steps_for(self, column: str) -> list[PlanStep]:
        return [s for s in self.steps if s.column == column]


class DecisionEntry(BaseModel):
    stage: str
    actor: Literal["system", "fallback", "planner", "critic", "validator", "user", "writer"]
    decision: str
    reason: str = ""


class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Literal["error", "warning", "info"] = "error"
    message: str
    details: dict = {}


class ValidationReport(BaseModel):
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]
