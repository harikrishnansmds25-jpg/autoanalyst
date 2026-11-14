from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from autoanalyst.schemas import DecisionEntry


class Stage(str, Enum):
    UPLOADED = "uploaded"
    PLANNING = "planning"
    AWAITING_PLAN = "awaiting_plan"      # human checkpoint 1
    EXECUTING = "executing"
    AWAITING_MODEL = "awaiting_model"    # human checkpoint 2
    TRAINING = "training"
    REPORTING = "reporting"
    DONE = "done"
    FAILED = "failed"


WORKING = {Stage.PLANNING, Stage.EXECUTING, Stage.TRAINING, Stage.REPORTING}


class InvalidAction(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobState(BaseModel):
    job_id: str
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    stage: Stage = Stage.UPLOADED
    message: str = ""
    error: str | None = None
    busy: bool = False

    file_name: str
    data_hash: str
    n_rows: int
    n_cols: int
    target: str

    plan_status: str | None = None
    open_issues: list[str] = []
    critic_remarks: list[str] = []
    execution_explanation: str | None = None
    suggested_task: str | None = None
    task: str | None = None
    model_key: str | None = None

    decision_log: list[DecisionEntry] = []
    timings: dict[str, float] = {}
