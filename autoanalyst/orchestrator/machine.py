"""The orchestrator: a plain state machine over one job folder. No LLM decides what runs next.

    UPLOADED -> PLANNING -> AWAITING_PLAN (checkpoint 1) -> EXECUTING -> AWAITING_MODEL (checkpoint 2)
             -> TRAINING -> REPORTING -> DONE

Every transition is saved to runs/<job_id>/state.json, so a job survives UI reruns and restarts.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import joblib

from autoanalyst.agents.cache import PlanCache, cache_key
from autoanalyst.agents.debate import DebateResult, ExecutionOutcome, execute_with_review, plan_with_debate
from autoanalyst.agents.writer import write_narrative
from autoanalyst.config import Settings, get_settings
from autoanalyst.eda import EdaResult, run_eda
from autoanalyst.ingest import LoadedData, load_csv
from autoanalyst.llm.client import LLMClient, LLMError, make_client
from autoanalyst.models import get_model
from autoanalyst.orchestrator.jobs import job_logger, new_job_id, read_state
from autoanalyst.orchestrator.manifest import llm_summary, public_config, versions
from autoanalyst.orchestrator.state import InvalidAction, JobState, Stage, now
from autoanalyst.profiling import build_profile
from autoanalyst.report import build_facts, build_report
from autoanalyst.schemas import CheckResult, DecisionEntry, Plan, Profile, ValidationReport
from autoanalyst.toolbox.executor import execute
from autoanalyst.training import TrainingError, TrainResult, train_and_evaluate
from autoanalyst.validators import check_plan
from autoanalyst.viz import make_charts


class PlanNotValid(ValueError):
    def __init__(self, errors: list[CheckResult]):
        self.errors = errors
        super().__init__("; ".join(e.message for e in errors))


class Job:
    def __init__(self, job_dir: Path, settings: Settings | None = None):
        self.dir = Path(job_dir)
        self.settings = settings or get_settings()
        state = read_state(self.dir)
        if state is None:
            raise FileNotFoundError(f"No job in {self.dir}")
        self.state = state
        self.log = job_logger(self.dir)

    @classmethod
    def create(cls, raw: bytes, file_name: str, settings: Settings | None = None,
               target: str | None = None) -> "Job":
        """Validate and profile an upload. Raises IngestError (nothing is kept) if the file is unusable."""
        settings = settings or get_settings()
        data = load_csv(raw, file_name, settings)
        profile = build_profile(data, target, settings)
        job_dir = Path(settings.runs_dir) / new_job_id()
        job_dir.mkdir(parents=True)
        (job_dir / "upload.csv").write_bytes(raw)
        (job_dir / "profile.json").write_text(profile.model_dump_json(indent=1), encoding="utf-8")
        state = JobState(
            job_id=job_dir.name, file_name=file_name, data_hash=data.data_hash,
            n_rows=len(data.df), n_cols=data.df.shape[1], target=profile.target,
            message="File checked and profiled. Confirm the target to start planning.",
        )
        (job_dir / "state.json").write_text(state.model_dump_json(indent=1), encoding="utf-8")
        job = cls(job_dir, settings)
        job.log.info("created from %s (%d rows, %d columns)", file_name, state.n_rows, state.n_cols)
        return job

    @classmethod
    def open(cls, job_id: str, settings: Settings | None = None) -> "Job":
        settings = settings or get_settings()
        return cls(Path(settings.runs_dir) / job_id, settings)

    # persistence

    def save(self) -> None:
        self.state.updated_at = now()
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(self.state.model_dump_json(indent=1), encoding="utf-8")
        tmp.replace(self.dir / "state.json")

    def reload(self) -> None:
        self.state = read_state(self.dir) or self.state

    def write_json(self, name: str, data) -> None:
        text = data.model_dump_json(indent=1) if hasattr(data, "model_dump_json") else json.dumps(data, indent=1, default=str)
        (self.dir / name).write_text(text, encoding="utf-8")

    def read_json(self, name: str, model=None):
        path = self.dir / name
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return model.model_validate_json(text) if model else json.loads(text)

    def data(self) -> LoadedData:
        return load_csv((self.dir / "upload.csv").read_bytes(), self.state.file_name, self.settings)

    def profile(self) -> Profile:
        return self.read_json("profile.json", Profile)

    def proposed_plan(self) -> Plan | None:
        return self.read_json("plan_proposed.json", Plan)

    def approved_plan(self) -> Plan | None:
        return self.read_json("plan_approved.json", Plan)

    def eda(self) -> EdaResult | None:
        return self.read_json("eda.json", EdaResult)

    def validation(self) -> ValidationReport | None:
        return self.read_json("validation.json", ValidationReport)

    def train_result(self) -> TrainResult | None:
        return self.read_json("train_result.json", TrainResult)

    def report_path(self, kind: str) -> Path | None:
        path = self.dir / f"report.{kind}"
        return path if path.exists() else None

    # helpers

    def _require(self, *stages: Stage) -> None:
        if self.state.stage not in stages:
            allowed = ", ".join(s.value for s in stages)
            raise InvalidAction(f"Job is '{self.state.stage.value}'; this needs: {allowed}")

    def _enter(self, stage: Stage, message: str, error: str | None = None) -> None:
        self.state.stage, self.state.message, self.state.error = stage, message, error
        self.save()
        self.log.info("stage %s: %s%s", stage.value, message, f" ({error})" if error else "")

    def _decide(self, *entries: DecisionEntry) -> None:
        self.state.decision_log.extend(entries)
        self.save()

    @contextmanager
    def _timed(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.state.timings[name] = round(self.state.timings.get(name, 0.0) + time.perf_counter() - started, 2)
            self.save()

    def _client(self) -> LLMClient | None:
        try:
            return make_client(self.settings)
        except LLMError as exc:
            self._decide(DecisionEntry(stage="setup", actor="system", decision="LLM unavailable",
                                       reason=f"{exc}. Rule-based fallbacks are used."))
            return None

    def _record_calls(self, client: LLMClient | None) -> None:
        if client is None or not client.calls:
            return
        calls = (self.read_json("llm_calls.json") or []) + client.call_log()
        self.write_json("llm_calls.json", calls)

    def _cache(self) -> PlanCache:
        return PlanCache(self.settings.cache_dir or Path(self.settings.runs_dir) / "_cache")

    def fail(self, message: str) -> None:
        self.state.busy = False
        self._enter(Stage.FAILED, "The job stopped because of an unexpected error.", message)

    # transitions

    def set_target(self, target: str) -> None:
        """Change the target before planning; the profile is rebuilt for it."""
        self._require(Stage.UPLOADED)
        if target == self.state.target:
            return
        profile = build_profile(self.data(), target, self.settings)
        self.write_json("profile.json", profile)
        self.state.target = target
        self._decide(DecisionEntry(stage="upload", actor="user", decision=f"Target set to '{target}'"))

    def plan(self, feedback: list[str] | None = None) -> DebateResult:
        """Planner-Critic debate. Ends at checkpoint 1 whatever the outcome."""
        self._require(Stage.UPLOADED, Stage.PLANNING)
        self._enter(Stage.PLANNING, "Planner and Critic are preparing the cleaning plan.")
        with self._timed("planning"):
            client = self._client()
            profile = self.profile()
            result = plan_with_debate(
                profile, client, self.settings, use_critic=self.settings.use_critic, cache=self._cache(),
                previous=self.proposed_plan() if feedback else None, feedback=feedback,
            )
            self._record_calls(client)
        self.write_json("plan_proposed.json", result.plan)
        debates = self.read_json("debate.json") or []
        debates.append({
            "status": result.status,
            "rounds": [
                {"number": r.number, "notes": r.notes, "plan": r.plan.model_dump(),
                 "validator": [c.model_dump() for c in r.checks],
                 "critic": r.review.model_dump() if r.review else None}
                for r in result.rounds
            ],
            "open_issues": result.open_issues,
        })
        self.write_json("debate.json", debates)
        last_review = result.rounds[-1].review if result.rounds else None
        self.state.plan_status = result.status
        self.state.open_issues = result.open_issues
        self.state.critic_remarks = (
            [f"{i.severity}: {i.column or 'table'}: {i.problem} {i.suggestion}" for i in last_review.issues]
            + [last_review.summary] if last_review else []
        )
        self._decide(*result.log)
        messages = {
            "approved": "The plan was approved by the Critic and the validators. Review it below.",
            "cached": "A plan approved earlier for this data and configuration was reused.",
            "fallback": "The rule-based plan is shown because the LLM was not available.",
            "escalated": "Planner and Critic did not agree. Review the open issues and edit the plan.",
        }
        self._enter(Stage.AWAITING_PLAN, messages[result.status])
        return result

    def check(self, plan: Plan) -> list[CheckResult]:
        return check_plan(plan, self.profile(), self.settings)

    def approve_plan(self, plan: Plan | None = None) -> None:
        """Checkpoint 1. A plan with validator errors cannot be approved, edited or not."""
        self._require(Stage.AWAITING_PLAN)
        proposed = self.proposed_plan()
        plan = plan or proposed
        errors = [c for c in self.check(plan) if c.severity == "error"]
        if errors:
            raise PlanNotValid(errors)
        edited = [s.model_dump() for s in plan.steps] != [s.model_dump() for s in proposed.steps]
        if edited:
            plan = plan.model_copy(update={"source": "user"})
        self.write_json("plan_approved.json", plan)
        if self.state.plan_status != "cached":
            self._cache().put(cache_key(self.profile(), self.settings,
                                        "critic" if self.settings.use_critic else "no_critic"), plan)
        self.state.execution_explanation = None
        self._decide(DecisionEntry(stage="plan", actor="user",
                                   decision="Plan edited and approved" if edited else "Plan approved",
                                   reason=f"{len(plan.steps)} steps"))
        self._enter(Stage.EXECUTING, "Running the approved plan and the validators.")

    def reject_plan(self, comment: str) -> list[str]:
        """Checkpoint 1 rejection: the comment becomes Planner feedback for a new debate."""
        self._require(Stage.AWAITING_PLAN)
        comment = comment.strip() or "The user rejected the plan without a comment."
        self._decide(DecisionEntry(stage="plan", actor="user", decision="Plan rejected", reason=comment[:300]))
        self._enter(Stage.PLANNING, "Re-planning with your feedback.")
        return [f"User: {comment}"]

    def execute(self) -> ExecutionOutcome:
        """Execute, validate, let the Critic review and the Planner fix (bounded), then run EDA."""
        self._require(Stage.EXECUTING)
        with self._timed("execution"):
            client = self._client()
            data, profile = self.data(), self.profile()
            outcome = execute_with_review(data.df, self.approved_plan(), profile, client, self.settings,
                                          use_critic=self.settings.use_critic)
            self._record_calls(client)
        self.write_json("validation.json", outcome.report)
        fixes = self.read_json("fix_rounds.json") or []
        fixes.append([
            {"number": r.number, "plan": r.plan.model_dump(),
             "validator": [c.model_dump() for c in r.report.checks],
             "critic": r.review.model_dump() if r.review else None}
            for r in outcome.rounds
        ])
        self.write_json("fix_rounds.json", fixes)
        self._decide(*outcome.log)

        if outcome.status == "stopped":
            self.write_json("plan_proposed.json", outcome.plan)
            self.state.execution_explanation = outcome.explanation
            self.state.plan_status = "failed_validation"
            self._enter(Stage.AWAITING_PLAN, "Validation failed. Edit the plan and approve it again.",
                        outcome.explanation)
            return outcome

        if outcome.plan != self.approved_plan():
            self.write_json("plan_approved.json", outcome.plan)
        with self._timed("eda"):
            eda = run_eda(outcome.result.df, self.state.target, settings=self.settings)
        self.write_json("eda.json", eda)
        self.state.suggested_task = eda.target.task
        self._enter(Stage.AWAITING_MODEL, "Data cleaned and validated. Choose the target, task and model.")
        return outcome

    def select_model(self, target: str, task: str, model_key: str) -> str:
        """Checkpoint 2. Returns 'train', or 'replan' when the target changed (plan depends on it)."""
        self._require(Stage.AWAITING_MODEL)
        if target != self.state.target:
            profile = build_profile(self.data(), target, self.settings)
            self.write_json("profile.json", profile)
            self.state.target = target
            self._decide(DecisionEntry(stage="model", actor="user", decision=f"Target changed to '{target}'",
                                       reason="The plan is rebuilt and must be approved again"))
            self._enter(Stage.UPLOADED, "Target changed; a new plan is needed.")
            return "replan"
        spec = get_model(task, model_key)
        self.state.task, self.state.model_key = task, model_key
        self._decide(DecisionEntry(
            stage="model", actor="user", decision=f"{spec.label} for {task}",
            reason="Suggested task confirmed" if task == self.state.suggested_task
            else f"Task changed from suggested {self.state.suggested_task}"))
        self._enter(Stage.TRAINING, f"Training {spec.label}.")
        return "train"

    def train_and_report(self) -> None:
        self._require(Stage.TRAINING)
        plan, profile = self.approved_plan(), self.profile()
        result = execute(self.data().df, plan)
        try:
            with self._timed("training"):
                out = train_and_evaluate(result.df, self.state.target, self.state.task, self.state.model_key,
                                         result.steps_by_feature(), self.settings)
        except TrainingError as exc:
            self._enter(Stage.AWAITING_MODEL, "Training is not possible with these choices.", str(exc))
            return
        joblib.dump(out.pipeline, self.dir / "model.joblib")
        self.write_json("train_result.json", out.result)
        self._decide(DecisionEntry(
            stage="model", actor="system",
            decision="Beats the baseline" if out.result.beats_baseline else "Does not clearly beat the baseline",
            reason=f"{out.result.comparison_metric}: model {out.result.test[out.result.comparison_metric]}, "
                   f"baseline {out.result.test_baseline[out.result.comparison_metric]}"))

        self._enter(Stage.REPORTING, "Writing the report.")
        with self._timed("report"):
            eda = run_eda(result.df, self.state.target, self.state.task, self.settings)
            self.write_json("eda.json", eda)
            charts = make_charts(result.df, profile, eda, out, self.dir / "charts")
            facts = build_facts(profile, plan, self.validation(), len(result.df), eda, out.result,
                                self.state.decision_log, self._run_info())
            client = self._client()
            written = write_narrative(facts, client, self.settings)
            self._record_calls(client)
            self._decide(*written.log)
            facts = facts.model_copy(update={"decision_log": self.state.decision_log, "run": {
                **facts.run, **self._run_info()}})
            files = build_report(facts, written.narrative, charts, self.dir,
                                 convert_pdf=self.settings.convert_pdf, settings=self.settings)
        self.write_json("writer.json", {"source": written.source, "attempts": written.attempts,
                                        "rejected": written.rejected,
                                        "removed_sentences": files.removed_sentences})
        if files.removed_sentences:
            self._decide(DecisionEntry(stage="report", actor="system", decision="Unverified sentences removed",
                                       reason=str(sum(len(v) for v in files.removed_sentences.values()))))
        self.write_manifest()
        if files.pdf is None and self.settings.convert_pdf:
            self._enter(Stage.DONE, "Report ready (DOCX only).", f"PDF conversion failed: {files.pdf_error}")
        else:
            self._enter(Stage.DONE, "Report ready.")

    def _run_info(self) -> dict:
        llm = llm_summary(self.read_json("llm_calls.json") or [])
        return {
            "job_id": self.state.job_id,
            "plan_status": self.state.plan_status,
            "llm_provider": self.settings.llm_provider,
            "llm_calls": llm["calls"],
            "llm_failed_calls": llm["failed_calls"],
            "timings_seconds": dict(self.state.timings),
        }

    def write_manifest(self) -> dict:
        train = self.train_result()
        manifest = {
            "job_id": self.state.job_id,
            "created_at": self.state.created_at,
            "written_at": now(),
            "data": {"file_name": self.state.file_name, "data_hash": self.state.data_hash,
                     "n_rows": self.state.n_rows, "n_cols": self.state.n_cols},
            "seed": self.settings.seed,
            "config": public_config(self.settings),
            "libraries": versions(),
            "target": self.state.target,
            "task": self.state.task,
            "model": {"key": self.state.model_key, "best_params": train.best_params if train else None},
            "plan_status": self.state.plan_status,
            "plan": (self.approved_plan() or self.proposed_plan()).model_dump(),
            "decision_log": [e.model_dump() for e in self.state.decision_log],
            "agent_messages": {
                "debates": self.read_json("debate.json") or [],
                "fix_rounds": self.read_json("fix_rounds.json") or [],
                "writer": self.read_json("writer.json"),
            },
            "validation": [c.model_dump() for c in (self.validation() or ValidationReport(checks=[])).checks],
            "timings_seconds": self.state.timings,
            "llm": llm_summary(self.read_json("llm_calls.json") or []),
            "llm_calls": self.read_json("llm_calls.json") or [],
        }
        self.write_json("manifest.json", manifest)
        return manifest

