import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from conftest import make_messy_frame
from docx import Document

from autoanalyst.config import Settings
from autoanalyst.ingest import IngestError
from autoanalyst.llm import LLMClient, MockProvider
from autoanalyst.orchestrator import InvalidAction, Job, JobBusy, JobRunner, PlanNotValid, Stage
from autoanalyst.orchestrator.jobs import cleanup_runs, read_state
from autoanalyst.report.pdf import find_soffice
from refdata import DATA

FAST = dict(cv_folds=3, search_iter=2, n_jobs=1, importance_repeats=2, convert_pdf=False)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, llm_provider="mock", runs_dir=tmp_path / "runs", **FAST)


@pytest.fixture
def runner(settings):
    return JobRunner(settings, inline=True)


def _messy_job(settings, target="churn"):
    return Job.create(make_messy_frame().to_csv(index=False).encode(), "messy.csv", settings, target)


def _run_to_done(job, runner, task="classification", model="decision_tree"):
    runner.submit(job.state.job_id, "plan")
    job.reload()
    job.approve_plan()
    runner.submit(job.state.job_id, "execute")
    job.reload()
    job.select_model(job.state.target, task, model)
    runner.submit(job.state.job_id, "train_and_report")
    job.reload()
    return job


def test_create_validates_and_profiles(settings):
    job = _messy_job(settings)
    assert job.state.stage == Stage.UPLOADED and job.state.target == "churn"
    assert (job.dir / "upload.csv").exists() and job.profile().n_rows == 205
    with pytest.raises(IngestError):
        Job.create(b"a,b\n1,2\n", "tiny.csv", settings)
    assert len(list(settings.runs_dir.iterdir())) == 1


def test_full_run_with_both_checkpoints(settings, runner):
    job = _run_to_done(_messy_job(settings), runner)
    assert job.state.stage == Stage.DONE, job.state.error
    for name in ("report.docx", "facts.json", "manifest.json", "model.joblib", "eda.json",
                 "validation.json", "debate.json", "llm_calls.json", "job.log"):
        assert (job.dir / name).exists(), name
    actors = {e.actor for e in job.state.decision_log}
    assert {"planner", "critic", "user", "validator", "writer"} <= actors
    assert set(job.state.timings) >= {"planning", "execution", "training", "report"}
    assert not job.state.busy


def test_manifest_contents(settings, runner):
    job = _run_to_done(_messy_job(settings), runner)
    manifest = json.loads((job.dir / "manifest.json").read_text())
    assert manifest["data"]["data_hash"] == job.state.data_hash
    assert manifest["seed"] == 42
    assert manifest["libraries"]["scikit-learn"] and manifest["libraries"]["streamlit"]
    assert "gemini_api_key" not in manifest["config"]
    assert manifest["plan"]["steps"]
    assert manifest["agent_messages"]["debates"][0]["rounds"][0]["critic"]["verdict"] == "approve"
    assert manifest["llm"]["calls"] == len(manifest["llm_calls"]) > 0
    assert manifest["llm"]["calls_by_agent"].keys() >= {"planner", "critic", "writer"}
    assert manifest["timings_seconds"]["training"] > 0


def test_secret_never_written(tmp_path, runner):
    settings = Settings(_env_file=None, llm_provider="mock", gemini_api_key="TOPSECRET-123",
                        runs_dir=tmp_path / "runs", **FAST)
    job = _run_to_done(_messy_job(settings), JobRunner(settings, inline=True))
    for path in job.dir.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".log", ".csv"):
            assert "TOPSECRET" not in path.read_text(encoding="utf-8", errors="ignore"), path


def test_invalid_transitions_are_refused(settings, runner):
    job = _messy_job(settings)
    with pytest.raises(InvalidAction):
        job.approve_plan()
    with pytest.raises(InvalidAction):
        job.select_model("churn", "classification", "decision_tree")
    runner.submit(job.state.job_id, "plan")
    job.reload()
    with pytest.raises(InvalidAction):
        job.train_and_report()


def test_edited_plan_is_checked_and_marked_as_user(settings, runner):
    job = _messy_job(settings)
    runner.submit(job.state.job_id, "plan")
    job.reload()
    plan = job.proposed_plan()
    broken = plan.model_copy(update={"steps": [s for s in plan.steps if s.action != "encode_onehot"]})
    with pytest.raises(PlanNotValid, match="never encoded"):
        job.approve_plan(broken)
    assert job.state.stage == Stage.AWAITING_PLAN
    edited = plan.model_copy(update={"steps": [s for s in plan.steps if s.action != "cap_outliers"]})
    job.approve_plan(edited)
    assert job.approved_plan().source == "user"
    assert job.state.decision_log[-1].decision == "Plan edited and approved"


def test_rejection_feeds_the_comment_to_the_planner(settings):
    provider = MockProvider()
    job = _messy_job(settings)
    job._client = lambda: LLMClient(provider)
    job.plan()
    feedback = job.reject_plan("Keep the notes column out, and do not cap age.")
    job.plan(feedback=feedback)
    last_planner_prompt = [p for a, _, p in provider.prompts if a == "planner"][-1]
    assert "do not cap age" in last_planner_prompt and "<previous_plan>" in last_planner_prompt
    assert job.state.stage == Stage.AWAITING_PLAN


def test_target_change_at_checkpoint_two_requires_a_new_plan(settings, runner):
    job = _messy_job(settings)
    runner.submit(job.state.job_id, "plan")
    job.reload()
    job.approve_plan()
    runner.submit(job.state.job_id, "execute")
    job.reload()
    assert job.select_model("age", "regression", "ridge") == "replan"
    assert job.state.stage == Stage.UPLOADED and job.profile().target == "age"
    runner.submit(job.state.job_id, "plan")
    job.reload()
    target_steps = job.proposed_plan().steps_for("age")
    assert [s.action for s in target_steps] == ["keep"]


def test_validation_failure_returns_to_checkpoint_one(settings, runner):
    job = _messy_job(settings)
    runner.submit(job.state.job_id, "plan")
    job.reload()
    plan = job.proposed_plan()
    no_dedup = plan.model_copy(update={"steps": [s for s in plan.steps if s.action != "drop_duplicates"]})
    job.approve_plan(no_dedup)
    job._client = lambda: None
    job.execute()
    assert job.state.stage == Stage.AWAITING_PLAN
    assert "duplicate" in job.state.execution_explanation
    assert job.state.plan_status == "failed_validation"


def test_training_error_returns_to_checkpoint_two(settings, runner):
    job = _messy_job(settings, target="city")
    runner.submit(job.state.job_id, "plan")
    job.reload()
    job.approve_plan()
    runner.submit(job.state.job_id, "execute")
    job.reload()
    job.select_model("city", "regression", "ridge")
    runner.submit(job.state.job_id, "train_and_report")
    job.reload()
    assert job.state.stage == Stage.AWAITING_MODEL
    assert "not numeric" in job.state.error


def test_no_llm_uses_fallbacks_and_still_finishes(tmp_path):
    settings = Settings(_env_file=None, llm_provider="hosted", runs_dir=tmp_path / "runs", **FAST)
    job = _run_to_done(_messy_job(settings), JobRunner(settings, inline=True))
    assert job.state.stage == Stage.DONE
    assert job.state.plan_status == "fallback"
    decisions = [e.decision for e in job.state.decision_log]
    assert "LLM unavailable" in decisions and "Template narrative used" in decisions
    assert not (job.dir / "llm_calls.json").exists()


def test_approved_plan_is_cached_for_the_next_job(settings, runner):
    _run_to_done(_messy_job(settings), runner)
    second = _messy_job(settings)
    runner.submit(second.state.job_id, "plan")
    second.reload()
    assert second.state.plan_status == "cached"


def test_job_survives_reopening(settings, runner):
    job = _messy_job(settings)
    runner.submit(job.state.job_id, "plan")
    reopened = Job.open(job.state.job_id, settings)
    assert reopened.state.stage == Stage.AWAITING_PLAN
    assert reopened.proposed_plan() == Job.open(job.state.job_id, settings).proposed_plan()


def test_unexpected_errors_fail_the_job_and_are_logged(settings, runner, monkeypatch):
    job = _messy_job(settings)
    monkeypatch.setattr(Job, "plan", lambda self, **kw: 1 / 0)
    runner.submit(job.state.job_id, "plan")
    job.reload()
    assert job.state.stage == Stage.FAILED and "ZeroDivisionError" in job.state.error
    assert not job.state.busy
    assert "Traceback" in (job.dir / "job.log").read_text()


def test_background_runner(settings):
    runner = JobRunner(settings)
    job = _messy_job(settings)
    future = runner.submit(job.state.job_id, "plan")
    with pytest.raises(JobBusy):
        runner.submit(job.state.job_id, "plan")
    future.result(timeout=120)
    assert not runner.is_busy(job.state.job_id)
    job.reload()
    assert job.state.stage == Stage.AWAITING_PLAN and not job.state.busy


def test_retention_cleanup(settings, runner):
    old, recent = _messy_job(settings), _messy_job(settings)
    stale = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    old.state.updated_at = stale
    (old.dir / "state.json").write_text(old.state.model_dump_json())
    busy = _messy_job(settings)
    busy.state.updated_at, busy.state.busy = stale, True
    (busy.dir / "state.json").write_text(busy.state.model_dump_json())
    cache_file = settings.runs_dir / "_cache" / "old.json"
    cache_file.parent.mkdir()
    cache_file.write_text("{}")
    past = time.time() - 30 * 86400
    os.utime(cache_file, (past, past))

    removed = cleanup_runs(settings.runs_dir, retention_days=7)
    assert removed == [old.state.job_id]
    assert read_state(recent.dir) is not None and read_state(busy.dir) is not None
    assert not cache_file.exists()


@pytest.mark.skipif(find_soffice() is None, reason="LibreOffice not installed")
def test_full_run_on_iris_with_pdf(tmp_path):
    settings = Settings(_env_file=None, llm_provider="mock", runs_dir=tmp_path / "runs",
                        **{**FAST, "convert_pdf": True})
    job = Job.create((DATA / "iris.csv").read_bytes(), "iris.csv", settings)
    job = _run_to_done(job, JobRunner(settings, inline=True), "classification", "logistic_regression")
    assert job.state.stage == Stage.DONE and job.state.error is None
    assert job.report_path("pdf") is not None
    text = "\n".join(p.text for p in Document(job.report_path("docx")).paragraphs)
    assert "Plan approved" not in text  # the decision log is a table, not prose
    tables = Document(job.report_path("docx")).tables
    assert any("Plan approved" in c.text for t in tables for r in t.rows for c in r.cells)


def test_plan_table_roundtrip(settings, runner):
    from autoanalyst.ui.plan_table import ALL_ROWS, frame_plan, plan_frame

    job = _messy_job(settings)
    runner.submit(job.state.job_id, "plan")
    job.reload()
    plan = job.proposed_plan()
    frame = plan_frame(plan)
    assert ALL_ROWS in set(frame["column"])
    back, error = frame_plan(frame, "churn")
    assert error is None and [s.model_dump() for s in back.steps] == [s.model_dump() for s in plan.steps]
    frame.loc[len(frame)] = {"column": ALL_ROWS, "action": "impute_median", "reason": "",
                             "factor": None, "dayfirst": None}
    bad, error = frame_plan(frame, "churn")
    assert bad is None and "needs a column" in error
    frame = frame.iloc[:-1]
    frame.loc[len(frame)] = {"column": "age", "action": None, "reason": None, "factor": None, "dayfirst": None}
    skipped, error = frame_plan(frame, "churn")
    assert error is None and len(skipped.steps) == len(plan.steps)
